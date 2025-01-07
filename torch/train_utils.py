import math

import torch
import torch.nn as nn


class ScheduledOptim():
    """
    实现 Transformer 论文中的学习率调度器
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.d_model = d_model

    def step_and_update_lr(self):
        """步进优化器并更新学习率"""
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        """计算学习率缩放因子"""
        return min(
            math.sqrt(1.0 / max(1, self.current_steps)),
            math.sqrt(self.current_steps / max(1, self.warmup_steps ** 3))
        ) * math.sqrt(self.d_model)

    def _update_learning_rate(self):
        """更新学习率"""
        self.current_steps += 1
        lr = self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class LabelSmoothing(nn.Module):
    """
    标签平滑
    防止模型过于自信，提高泛化能力
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def training_tools_demo():
    """
    演示训练工具的使用
    """
    # 创建一个简单的模型
    model = nn.Linear(10, 5)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 创建学习率调度器
    scheduler = ScheduledOptim(
        optimizer=optimizer,
        d_model=512,  # 假设模型维度为512
        warmup_steps=4000
    )

    # 创建标签平滑
    criterion = LabelSmoothing(
        size=5,  # 输出维度
        padding_idx=0,  # padding的索引
        smoothing=0.1  # 平滑系数
    )

    # 模拟训练过程
    batch_size = 32
    for epoch in range(5):
        # 生成随机数据
        x = torch.randn(batch_size, 10)
        target = torch.randint(0, 5, (batch_size,))

        # 前向传播
        output = model(x)
        output = torch.log_softmax(output, dim=-1)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        scheduler.zero_grad()
        loss.backward()

        # 更新参数和学习率
        scheduler.step_and_update_lr()

        # 打印训练信息
        if epoch % 1 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, '
                  f'LR: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')


if __name__ == "__main__":
    training_tools_demo()
