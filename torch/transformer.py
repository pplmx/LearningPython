import torch
import torch.nn as nn


class SimpleTransformerLayer(nn.Module):
    """
    简单的 Transformer 层实现
    包含：
    1. 多头自注意力层
    2. Layer Normalization
    3. 前馈神经网络
    """

    def __init__(self, hidden_size=8, num_heads=2, dropout=0.1):
        super().__init__()

        # 多头注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # 第一个 Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # 第二个 Layer Normalization
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头自注意力 + 残差连接
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈神经网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


def transformer_demo():
    """演示简单 Transformer 层的使用"""
    # 设置随机种子
    torch.manual_seed(42)

    # 定义基本参数
    sequence_length = 10
    batch_size = 2
    hidden_size = 8

    # 创建模型
    model = SimpleTransformerLayer(hidden_size=hidden_size)

    # 创建输入
    input_tensor = torch.randn(sequence_length, batch_size, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # 前向传播
    output = model(input_tensor)
    print(f"\nOutput shape: {output.shape}")

    # 计算损失（这里用 MSE 损失作为示例）
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    # 反向传播
    loss.backward()

    # 打印模型参数和梯度信息
    print("\nModel parameters and gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"- Parameter shape: {param.shape}")
            print(f"- Gradient shape: {param.grad.shape}")


if __name__ == "__main__":
    transformer_demo()
