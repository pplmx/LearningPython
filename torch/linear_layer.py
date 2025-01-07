import torch
import torch.nn as nn


def linear_layer_demo():
    """
    演示 PyTorch 中 Linear 层的基本使用
    Linear 层进行的操作是: output = input @ weight.T + bias
    """
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)

    # 定义基本参数
    sequence_length = 10  # 序列长度
    batch_size = 2  # 批次大小
    hidden_size = 8  # 隐藏层大小（这里使用较小的值方便展示）

    # 创建输入张量: shape = (sequence_length, batch_size, hidden_size)
    # 在 Transformer 中，通常输入形状为 (sequence_length, batch_size, hidden_size)
    input_tensor = torch.randn(sequence_length, batch_size, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # 创建 Linear 层
    # in_features: 输入特征维度
    # out_features: 输出特征维度
    linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    # 查看 Linear 层的参数
    print(f"\nLinear layer weight shape: {linear.weight.shape}")  # (out_features, in_features)
    print(f"Linear layer bias shape: {linear.bias.shape}")  # (out_features,)

    # 前向传播
    output = linear(input_tensor)
    print(f"\nOutput shape: {output.shape}")

    # 创建模拟的梯度
    grad = torch.randn_like(output)

    # 反向传播
    output.backward(grad)

    # 打印权重的梯度
    print(f"\nWeight gradient shape: {linear.weight.grad.shape}")
    print(f"Bias gradient shape: {linear.bias.grad.shape}")

    # 打印具体的值（只显示部分）
    print("\nSample values:")
    print("Input (first position, first batch):")
    print(input_tensor[0, 0])
    print("\nOutput (first position, first batch):")
    print(output[0, 0])


if __name__ == '__main__':
    linear_layer_demo()
