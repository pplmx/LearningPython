import matplotlib.pyplot as plt

import torch
import torch.nn as nn

"""
torch.nn.LayerNorm的参数说明

主要参数:
1. normalized_shape:
   - 要在其上计算统计数据的特征形状
   - 可以是单个整数或多个维度的形状
   - 对于2D输入[batch_size, feature_dim]，设置为feature_dim
   - 对于3D输入[seq_len, batch_size, hidden_size]，通常设置为hidden_size
   - 对于4D输入[batch_size, channels, height, width]，可设置为[channels, height, width]

2. eps:
   - 默认值: 1e-5
   - 添加到分母的小常数，防止除零
   - 影响数值稳定性

3. elementwise_affine:
   - 默认值: True
   - 是否使用可学习的仿射参数gamma和beta
   - False时只进行标准化，不进行线性变换

4. device, dtype:
   - 用于指定参数的设备和数据类型
   - 通常会自动匹配输入张量
"""


def manual_layer_norm(x, eps=1e-5):
    """
    手动实现LayerNorm的计算过程

    Args:
        x (torch.Tensor): 输入张量，最后一个维度是要归一化的特征维度
        eps (float): 数值稳定性常数，默认1e-5

    Returns:
        torch.Tensor: 归一化后的张量，形状与输入相同

    Note:
        对于形状为[seq_len, batch_size, hidden_size]的输入：
        - 在hidden_size维度上计算均值和方差
        - 每个时间步(seq_len)和每个样本(batch_size)都独立归一化
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm


def plot_distribution(tensor, title):
    """
    绘制张量值的分布直方图

    Args:
        tensor (torch.Tensor): 要可视化的张量
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 4))
    plt.hist(tensor.flatten().detach().numpy(), bins=50)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_distributions(before, after):
    """
    并排比较归一化前后的分布

    Args:
        before (torch.Tensor): 归一化前的张量
        after (torch.Tensor): 归一化后的张量
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ax1.hist(before.flatten().detach().numpy(), bins=50)
    ax1.set_title("Before LayerNorm")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    ax2.hist(after.flatten().detach().numpy(), bins=50)
    ax2.set_title("After LayerNorm")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def layer_norm_demo():
    """
    演示 PyTorch 中 LayerNorm 层的基本使用

    输入张量维度说明:
    - seq_len: 序列长度维度 (第0维)
        * 表示序列中的位置或时间步
        * 在NLP任务中代表句子长度
        * 在Transformer中是token的位置

    - batch_size: 批次大小维度 (第1维)
        * 表示一次处理的样本数量
        * 每个样本都独立进行归一化
        * 批处理提高计算效率

    - hidden_size: 隐藏层大小维度 (第2维)
        * 表示每个位置的特征维度
        * LayerNorm在此维度上计算统计量
        * 在Transformer中是词嵌入或隐藏状态的维度

    LayerNorm特点:
    1. 计算过程:
       - 在最后一个维度(hidden_size)上计算均值和方差
       - 每个样本和每个位置独立进行归一化
       - y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    2. 与BatchNorm对比:
       - BatchNorm: 在batch维度上归一化，依赖batch size
       - LayerNorm: 在特征维度上归一化，独立于batch size
       - LayerNorm更适合NLP任务和小batch size场景

    3. 维度处理示例:
       input[seq_len=10, batch_size=2, hidden_size=8]
       - 对shape为8的最后一个维度归一化
       - 10*2=20个独立的归一化操作
       - 每个归一化操作处理8个特征值
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    print("=== 1. 基本示例 ===")
    # 定义输入张量的维度
    seq_len = 10  # 序列中的位置数量
    batch_size = 2  # 一次处理的样本数量
    hidden_size = 8  # 每个位置的特征数量

    # 创建输入张量 [seq_len, batch_size, hidden_size]
    input_tensor = torch.randn(seq_len, batch_size, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # 计算原始输入的统计信息
    print("\nBefore LayerNorm:")
    print(f"Mean: {input_tensor.mean():.4f}")
    print(f"Std: {input_tensor.std():.4f}")

    # 创建 LayerNorm 层
    # normalized_shape=hidden_size 表示在最后一个维度(大小为8)上进行归一化
    layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    # 查看 LayerNorm 层的参数
    print(f"\nLayerNorm parameters:")
    print(f"Weight (gamma) shape: {layer_norm.weight.shape}")  # shape: [hidden_size]
    print(f"Bias (beta) shape: {layer_norm.bias.shape}")  # shape: [hidden_size]

    # 前向传播
    output = layer_norm(input_tensor)

    # 计算归一化后的统计信息
    print("\nAfter LayerNorm:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")

    print("\n=== 2. 手动实现验证 ===")
    # 比较手动实现和PyTorch实现
    manual_output = manual_layer_norm(input_tensor)
    torch_output = nn.LayerNorm(hidden_size, elementwise_affine=False)(input_tensor)
    max_diff = (manual_output - torch_output).abs().max()
    print(f"Maximum difference between manual and torch implementation: {max_diff:.8f}")

    print("\n=== 3. 不同维度示例 ===")
    # 2D输入示例
    input_2d = torch.randn(32, 64)  # [batch_size, feature_dim]
    ln_2d = nn.LayerNorm(64)
    output_2d = ln_2d(input_2d)
    print(f"2D input shape: {input_2d.shape} -> output shape: {output_2d.shape}")

    # 4D输入示例
    input_4d = torch.randn(32, 16, 28, 28)  # [batch_size, channels, height, width]
    ln_4d = nn.LayerNorm([16, 28, 28])
    output_4d = ln_4d(input_4d)
    print(f"4D input shape: {input_4d.shape} -> output shape: {output_4d.shape}")

    print("\n=== 4. 反向传播示例 ===")
    grad = torch.randn_like(output)
    output.backward(grad)
    print(f"Weight gradient shape: {layer_norm.weight.grad.shape}")
    print(f"Bias gradient shape: {layer_norm.bias.grad.shape}")

    print("\n=== 5. 数值对比 ===")
    print("Sample values comparison (first position, first batch):")
    print("Before normalization:")
    print(input_tensor[0, 0])  # 第一个位置，第一个样本的所有特征
    print("\nAfter normalization:")
    print(output[0, 0])  # 归一化后的结果

    print("\n=== 6. 分布可视化 ===")
    print("Plotting distributions...")
    compare_distributions(input_tensor, output)


if __name__ == "__main__":
    layer_norm_demo()
