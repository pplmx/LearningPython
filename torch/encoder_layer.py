import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    包含:
    1. 多头自注意力机制
    2. 前馈神经网络
    3. Layer Normalization
    4. 残差连接
    """

    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈神经网络的隐藏层维度
            dropout: dropout比率
        """
        super().__init__()

        # 多头注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量，shape = (batch_size, seq_length, d_model)
            mask: 注意力掩码（可选）
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈神经网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


def encoder_layer_demo():
    """
    演示编码器层的使用
    """
    # 设置随机种子
    torch.manual_seed(42)

    # 设置参数
    batch_size = 2
    seq_length = 10
    d_model = 512

    # 创建编码器层
    encoder = EncoderLayer(
        d_model=d_model,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    )

    # 创建输入张量
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"Input shape: {x.shape}")

    # 创建掩码（可选）
    # 这里创建一个简单的填充掩码，假设序列后半部分是填充的
    mask = torch.ones(batch_size, seq_length)
    mask[:, seq_length // 2:] = 0
    print("\nMask shape:", mask.shape)
    print("Mask (1表示有效位置，0表示填充位置):")
    print(mask)

    # 前向传播
    output = encoder(x, mask)
    print(f"\nOutput shape: {output.shape}")

    # 检查输出的统计信息
    print("\nOutput statistics:")
    print(f"Mean: {output.mean().item():.4f}")
    print(f"Std: {output.std().item():.4f}")

    # 打印模型参数信息
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


if __name__ == '__main__':
    encoder_layer_demo()
