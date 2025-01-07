import torch
import torch.nn as nn


def layer_norm_demo():
    """
    演示 PyTorch 中 LayerNorm 层的基本使用
    LayerNorm 的作用是对输入进行归一化，使其均值为0，方差为1
    公式: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    其中 gamma 和 beta 是可学习的参数
    """
    # 设置随机种子
    torch.manual_seed(42)

    # 定义基本参数
    sequence_length = 10
    batch_size = 2
    hidden_size = 8

    # 创建输入张量
    input_tensor = torch.randn(sequence_length, batch_size, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # 计算原始输入的统计信息
    print("\nBefore LayerNorm:")
    print(f"Mean: {input_tensor.mean():.4f}")
    print(f"Std: {input_tensor.std():.4f}")

    # 创建 LayerNorm 层
    # normalized_shape: 需要归一化的维度的形状
    layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    # 查看 LayerNorm 层的参数
    print(f"\nLayerNorm weight (gamma) shape: {layer_norm.weight.shape}")
    print(f"LayerNorm bias (beta) shape: {layer_norm.bias.shape}")

    # 前向传播
    output = layer_norm(input_tensor)

    # 计算归一化后的统计信息
    print("\nAfter LayerNorm:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")

    # 创建模拟的梯度
    grad = torch.randn_like(output)

    # 反向传播
    output.backward(grad)

    # 打印梯度
    print(f"\nWeight gradient shape: {layer_norm.weight.grad.shape}")
    print(f"Bias gradient shape: {layer_norm.bias.grad.shape}")

    # 显示具体数值比较
    print("\nSample values comparison (first position, first batch):")
    print("Before normalization:")
    print(input_tensor[0, 0])
    print("\nAfter normalization:")
    print(output[0, 0])


if __name__ == '__main__':
    layer_norm_demo()
