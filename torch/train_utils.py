import math

import torch
import torch.nn as nn


class ScheduledOptim:
    """Transformer 的学习率调度器

    这个调度器实现了 Transformer 论文中的学习率调整策略:
    1. 在预热阶段(warmup_steps之前)逐渐增加学习率
    2. 在预热阶段之后逐渐降低学习率

    参数:
        optimizer: torch.optim - PyTorch优化器实例
        d_model: int - Transformer模型的维度
        warmup_steps: int - 预热步数,默认4000步
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_steps = 0  # 当前训练步数
        self.d_model = d_model

        # 打印初始设置
        print("初始化学习率调度器:")
        print(f"- 模型维度: {d_model}")
        print(f"- 预热步数: {warmup_steps}")
        print(f"- 初始学习率: {optimizer.param_groups[0]['lr']:.6f}")

    def step_and_update_lr(self):
        """执行优化器步进并更新学习率"""
        self._update_learning_rate()  # 先更新学习率
        self.optimizer.step()  # 再更新模型参数

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        """计算学习率的缩放因子

        实现公式: lr = min(1/√step, step/(warmup_steps^3)) * √d_model

        返回:
            float: 学习率缩放因子
        """
        d_scale = math.sqrt(self.d_model)  # 模型维度的缩放

        # 预热阶段前后使用不同的计算公式
        step_scale = min(
            math.sqrt(1.0 / max(1, self.current_steps)),  # 预热后的下降曲线
            math.sqrt(
                self.current_steps / max(1, self.warmup_steps**3)
            ),  # 预热阶段的上升曲线
        )

        return step_scale * d_scale

    def _update_learning_rate(self):
        """更新优化器中的学习率"""
        self.current_steps += 1
        lr = self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class LabelSmoothing(nn.Module):
    """标签平滑模块

    将one-hot标签转换为软标签,防止模型过于自信:
    1. 正确类别的概率从1变为confidence (< 1)
    2. 其他类别均分剩余概率(smoothing)

    参数:
        size: int - 类别数量
        padding_idx: int - 填充标签的索引(该位置不参与平滑)
        smoothing: float - 平滑系数,一般取小值如0.1
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")  # 使用KL散度作为损失函数
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 正确类别的概率
        self.smoothing = smoothing  # 平滑系数
        self.size = size  # 类别数量
        self.true_dist = None

        print("初始化标签平滑:")
        print(f"- 类别数量: {size}")
        print(f"- 平滑系数: {smoothing}")
        print(f"- 正确标签概率: {self.confidence}")
        print(f"- 其他标签概率: {smoothing / (size - 2):.4f}")

    def forward(self, x, target):
        """
        参数:
            x: torch.Tensor - 模型输出的对数概率 [batch_size, num_classes]
            target: torch.Tensor - 目标标签 [batch_size]

        返回:
            torch.Tensor - 平滑后的KL散度损失
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        # 1. 所有位置填充平滑概率
        true_dist.fill_(self.smoothing / (self.size - 2))

        # 2. 正确类别位置填充confidence概率
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 3. padding位置填充0
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def training_tools_demo():
    """演示学习率调度和标签平滑的使用

    通过一个简单的线性模型展示:
    1. 学习率如何随着训练步数变化
    2. 标签平滑如何软化目标概率分布
    """
    print("=" * 50)
    print("开始训练工具演示")
    print("=" * 50)

    # 1. 创建一个简单的线性模型
    input_size, output_size = 10, 5
    model = nn.Linear(input_size, output_size)
    print(f"\n模型结构:\n{model}")

    # 2. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 3. 创建学习率调度器
    scheduler = ScheduledOptim(
        optimizer=optimizer,
        d_model=512,  # Transformer模型维度
        warmup_steps=4000,
    )

    # 4. 创建标签平滑
    criterion = LabelSmoothing(
        size=output_size,  # 输出类别数
        padding_idx=0,  # padding的索引
        smoothing=0.1,  # 10%的平滑系数
    )

    # 5. 模拟训练过程
    print("\n开始训练:")
    batch_size = 32
    for epoch in range(5):
        # 生成随机数据
        x = torch.randn(batch_size, input_size)  # 随机输入
        target = torch.randint(0, output_size, (batch_size,))  # 随机标签

        # 前向传播
        output = model(x)
        output = torch.log_softmax(output, dim=-1)  # 转换为对数概率

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        scheduler.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度

        # 更新参数和学习率
        scheduler.step_and_update_lr()

        # 打印训练信息
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss.item():.4f}, "
            f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}"
        )


if __name__ == "__main__":
    training_tools_demo()
