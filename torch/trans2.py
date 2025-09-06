import math

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现

    参数:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 初始化权重矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(self.d_k)

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """计算缩放点积注意力，添加数值稳定性的处理

        参数:
            q: Query张量
            k: Key张量
            v: Value张量
            mask: 可选的掩码张量

        返回:
            attention输出
        """
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 为数值稳定性添加防护
        scores_max, _ = scores.detach().max(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - scores_max)

        if mask is not None:
            exp_scores = exp_scores.masked_fill(mask == 0, 0)

        # 计算注意力权重
        attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # 应用注意力权重
        return torch.matmul(attn_weights, v)

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """将输入张量分割成多个注意力头

        参数:
            x: 输入张量
            batch_size: 批次大小

        返回:
            重塑后的张量
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size = q.size(0)

        # 线性变换并分割注意力头
        Q = self.split_heads(self.W_q(q), batch_size)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(self.W_k(k), batch_size)
        V = self.split_heads(self.W_v(v), batch_size)

        # 计算注意力
        attn_output = self.attention(Q, K, V, mask)

        # 合并注意力头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


class FeedForward(nn.Module):
    """前馈神经网络实现"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        # 自注意力
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 交叉注意力
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))


class PositionalEncoding(nn.Module):
    """位置编码实现"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    """完整的Transformer模型实现"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # 词嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        # 输出层
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self) -> None:
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """生成用于解码器的方形后续掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """编码器前向传播"""
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        return enc_output

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """解码器前向传播"""
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, memory, src_mask, tgt_mask)
        return dec_output

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """模型前向传播"""
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        return self.fc(dec_output)


def create_mask(src: Tensor, tgt: Tensor, pad_idx: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """创建源序列和目标序列的掩码

    参数:
        src: 源序列
        tgt: 目标序列
        pad_idx: 填充标记的索引
        device: 计算设备

    返回:
        src_mask: 源序列掩码
        tgt_mask: 目标序列掩码
    """
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)

    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()

    tgt_mask = tgt_mask & ~subsequent_mask

    return src_mask, tgt_mask


def demo() -> None:
    """运行Transformer模型的演示，添加详细的调试信息"""
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型参数 - 使用更小的维度便于演示
    params = {
        "src_vocab_size": 10,
        "tgt_vocab_size": 10,
        "d_model": 32,  # 进一步减小模型维度
        "num_heads": 2,  # 减少注意力头数
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 64,  # 减小前馈网络维度
        "max_seq_length": 100,
        "dropout": 0.1,
    }

    print("\nModel Parameters:", params)

    # 创建模型
    model = Transformer(**params).to(device)
    model.train()

    # 打印模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # 创建优化器，使用更小的学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 创建示例数据
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
    tgt = torch.tensor([[0, 2, 3, 4], [0, 6, 7, 8]], device=device)
    tgt_true = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]], device=device)

    print("\nInput shapes:")
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")

    # 创建掩码
    src_mask, tgt_mask = create_mask(src, tgt, pad_idx=0, device=device)

    def check_grad_norm():
        """检查梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    print("\n开始训练...")
    for epoch in range(10):  # 增加训练轮数
        optimizer.zero_grad()

        # 前向传播
        output = model(src, tgt, src_mask, tgt_mask)

        # 检查输出是否包含NaN
        if torch.isnan(output).any():
            print(f"Warning: NaN detected in output at epoch {epoch + 1}")
            continue

        # 计算损失
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        loss = criterion(output.reshape(-1, params["tgt_vocab_size"]), tgt_true.reshape(-1))

        # 检查损失是否为NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch + 1}")
            continue

        # 反向传播
        loss.backward()

        # 检查梯度
        grad_norm = check_grad_norm()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 使用更小的max_norm

        optimizer.step()

        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm (before clipping): {grad_norm:.4f}")

        # 检查一些层的权重范数
        with torch.no_grad():
            enc_norm = model.encoder_embedding.weight.norm().item()
            print(f"  Encoder embedding norm: {enc_norm:.4f}")

    # 评估模式
    model.eval()
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
        predictions = output.argmax(dim=-1)

        # 检查logits的范围
        print("\nLogits statistics:")
        print(f"Min logit: {output.min().item():.4f}")
        print(f"Max logit: {output.max().item():.4f}")
        print(f"Mean logit: {output.mean().item():.4f}")

        print("\n评估结果:")
        print(f"输入序列:\n{src}")
        print(f"\n目标序列:\n{tgt_true}")
        print(f"\n预测序列:\n{predictions}")
        print(f"\n输出logits形状: {output.shape}")


if __name__ == "__main__":
    demo()
