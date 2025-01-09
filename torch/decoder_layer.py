import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    包含:
    1. 带掩码的多头自注意力机制
    2. 编码器-解码器交叉注意力
    3. 前馈神经网络
    4. Layer Normalization
    5. 残差连接
    """

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # 第一个多头自注意力层（带掩码）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 第二个多头注意力层（用于编码器-解码器交叉注意力）
        self.cross_attn = nn.MultiheadAttention(
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

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        参数:
            x: 解码器输入
            enc_output: 编码器输出
            tgt_mask: 目标序列掩码（防止看到未来信息）
            src_mask: 源序列掩码（用于处理填充）
        """
        # 自注意力
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 交叉注意力
        cross_attn_out, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


def decoder_layer_demo():
    """
    演示解码器层的使用
    展示:
    1. 解码器的输入处理
    2. 掩码的创建和应用
    3. 与编码器输出的交互
    """
    # 设置随机种子
    torch.manual_seed(42)

    # 设置参数
    batch_size = 2
    seq_length = 10
    d_model = 512

    # 创建解码器层
    decoder = DecoderLayer(d_model=d_model, num_heads=8, d_ff=2048, dropout=0.1)

    # 创建模拟输入
    decoder_input = torch.randn(batch_size, seq_length, d_model)  # 解码器输入
    encoder_output = torch.randn(batch_size, seq_length, d_model)  # 编码器输出
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")

    # 创建目标序列掩码（防止看到未来信息）
    tgt_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    print("\nTarget mask shape:", tgt_mask.shape)
    print("Target mask (上三角为-inf，表示不能看到未来信息):")
    print(tgt_mask[:5, :5])  # 只打印一部分

    # 创建源序列掩码（用于处理填充）
    src_mask = torch.ones(batch_size, seq_length)
    src_mask[:, seq_length // 2:] = 0  # 假设后半部分是填充
    print("\nSource mask shape:", src_mask.shape)
    print("Source mask (1表示有效位置，0表示填充位置):")
    print(src_mask)

    # 前向传播
    output = decoder(decoder_input, encoder_output, tgt_mask, src_mask)
    print(f"\nOutput shape: {output.shape}")

    # 检查输出的统计信息
    print("\nOutput statistics:")
    print(f"Mean: {output.mean().item():.4f}")
    print(f"Std: {output.std().item():.4f}")

    # 打印模型参数信息
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


if __name__ == '__main__':
    decoder_layer_demo()
