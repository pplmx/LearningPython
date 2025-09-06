import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    位置编码器
    用于为输入序列添加位置信息，因为 Transformer 的结构本身不包含位置信息
    使用正弦和余弦函数的组合来生成位置编码
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        参数:
            d_model: 模型的维度
            max_seq_length: 最大序列长度
        """
        super().__init__()

        # 创建位置编码矩阵 (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # 创建位置矩阵 (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # 创建除数矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos

        # 添加batch维度 (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)

        # 将位置编码注册为缓冲区（不参与训练）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量，形状为 (batch_size, seq_length, d_model)
        返回:
            添加了位置编码的张量，形状与输入相同
        """
        return x + self.pe[:, : x.size(1)]


def visualize_positional_encoding():
    """
    可视化位置编码矩阵
    """
    # 创建位置编码器
    d_model = 64
    max_seq_length = 100
    pos_encoder = PositionalEncoding(d_model, max_seq_length)

    # 获取位置编码矩阵
    pe = pos_encoder.pe.squeeze(0).numpy()

    # 创建热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(pe, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Positional Encoding Values")
    plt.show()


def positional_encoding_demo():
    """
    演示位置编码的使用
    """
    # 设置参数
    batch_size = 2
    seq_length = 10
    d_model = 8

    # 创建位置编码器
    pos_encoder = PositionalEncoding(d_model)

    # 创建输入张量
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"Input shape: {x.shape}")
    print("\nInput (first sequence):")
    print(x[0])

    # 应用位置编码
    output = pos_encoder(x)
    print(f"\nOutput shape: {output.shape}")
    print("\nOutput (first sequence):")
    print(output[0])

    # 展示位置编码的影响
    print("\nPositional encoding contribution (output - input):")
    print((output - x)[0])


if __name__ == "__main__":
    # 运行演示
    positional_encoding_demo()

    # 可视化位置编码
    visualize_positional_encoding()
