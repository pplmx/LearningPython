"""
Python单例模式的多种实现方式
每种方式都有不同的特点和适用场景
"""


# 1. Borg模式（共享状态单例）- 使用 __init__ 实现
class BorgSingleton:
    """
    Borg模式：实例不同，但状态共享
    特点：
    - 允许创建多个实例对象
    - 所有实例共享相同的状态（属性）
    - 通过 __init__ 方法实现状态共享
    """

    _shared_state = {}  # 类级别的共享状态字典

    def __init__(self):
        # 将实例的 __dict__ 指向共享状态字典
        self.__dict__ = self._shared_state


# 2. Borg模式（共享状态单例）- 使用 __new__ 实现
class BorgSingletonNew:
    """
    Borg模式的 __new__ 实现版本
    与上面的实现效果相同，但使用 __new__ 方法
    """

    _shared_state = {}

    def __new__(cls, *args, **kwargs):
        # 创建新实例
        obj = super().__new__(cls)
        # 将实例的 __dict__ 指向共享状态字典
        obj.__dict__ = cls._shared_state
        return obj


# 3. 经典单例模式 - 使用 __new__ 实现
class ClassicSingleton:
    """
    经典单例模式：确保只有一个实例
    特点：
    - 全局只创建一个实例对象
    - 多次实例化返回同一个对象
    - 使用 __new__ 方法控制实例创建
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


# 4. 元类单例模式
class SingletonMeta(type):
    """
    单例元类：通过元类实现单例模式
    特点：
    - 更加优雅的实现方式
    - 支持继承
    - 线程安全（在CPython中由于GIL）
    """

    _instances = {}  # 存储每个类的唯一实例

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # 创建新实例并存储
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class MetaSingleton(metaclass=SingletonMeta):
    """使用元类的单例类"""

    def __init__(self, value=None):
        self.value = value

    def display_value(self):
        return f"当前值: {self.value}"


# 5. 装饰器单例模式
def singleton(cls):
    """
    单例装饰器
    特点：
    - 简洁易用
    - 不修改原类结构
    - 适合简单场景
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class DecoratorSingleton:
    """使用装饰器的单例类"""

    def __init__(self, value=None):
        self.value = value

    def display_value(self):
        return f"装饰器单例值: {self.value}"


# 6. 线程安全的单例模式
import threading


class ThreadSafeSingleton:
    """
    线程安全的单例模式
    特点：
    - 使用双重检查锁定
    - 确保多线程环境下的安全性
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # 双重检查
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance


# 测试代码
def test_singletons():
    """测试各种单例模式的实现"""

    print("=" * 50)
    print("1. 测试 Borg 模式（共享状态）")
    print("-" * 30)

    borg1 = BorgSingleton()
    borg2 = BorgSingleton()

    borg1.name = "Borg实例"
    borg1.count = 42

    print(f"borg1 和 borg2 是同一个对象吗？{borg1 is borg2}")  # False
    print(f"borg1.name: {borg1.name}")
    print(f"borg2.name: {borg2.name}")  # 共享状态
    print(f"borg2.count: {borg2.count}")

    print("\n" + "=" * 50)
    print("2. 测试经典单例模式")
    print("-" * 30)

    classic1 = ClassicSingleton()
    classic2 = ClassicSingleton()

    classic1.value = "第一个实例"

    print(f"classic1 和 classic2 是同一个对象吗？{classic1 is classic2}")  # True
    print(f"classic1.value: {classic1.value}")
    print(f"classic2.value: {classic2.value}")  # 同一个对象

    print("\n" + "=" * 50)
    print("3. 测试元类单例模式")
    print("-" * 30)

    meta1 = MetaSingleton("元类实例")
    meta2 = MetaSingleton("另一个元类实例")

    print(f"meta1 和 meta2 是同一个对象吗？{meta1 is meta2}")  # True
    print(f"meta1.display_value(): {meta1.display_value()}")
    print(f"meta2.display_value(): {meta2.display_value()}")  # 会被覆盖

    print("\n" + "=" * 50)
    print("4. 测试装饰器单例模式")
    print("-" * 30)

    decorator1 = DecoratorSingleton("装饰器实例")
    decorator2 = DecoratorSingleton("另一个装饰器实例")

    print(f"decorator1 和 decorator2 是同一个对象吗？{decorator1 is decorator2}")  # True
    print(f"decorator1.display_value(): {decorator1.display_value()}")
    print(f"decorator2.display_value(): {decorator2.display_value()}")  # 同一个对象

    print("\n" + "=" * 50)
    print("5. 各种实现方式的比较")
    print("-" * 30)
    print("Borg模式: 实例不同，状态共享")
    print("经典单例: 实例相同，防止重复创建")
    print("元类单例: 优雅实现，支持继承")
    print("装饰器单例: 简洁易用，不修改类结构")
    print("线程安全单例: 适用于多线程环境")


if __name__ == "__main__":
    test_singletons()
