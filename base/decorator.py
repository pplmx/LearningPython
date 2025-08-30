"""
装饰器示例代码 - 演示四种不同的装饰器实现方式
包含函数式装饰器和类装饰器的实现
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ========== 函数式装饰器 ==========


def debug(func: F) -> F:
    """
    简单的调试装饰器 - 在函数调用前后打印消息
    用法: @debug
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"[DEBUG] 开始调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[DEBUG] 完成调用函数: {func.__name__}")
        return result

    return wrapper


def log(level: str) -> Callable[[F], F]:
    """
    带参数的日志装饰器 - 支持自定义日志级别
    用法: @log("INFO")

    Args:
        level: 日志级别标识，如 "INFO", "DEBUG", "ERROR" 等
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print(f"[{level}] 开始调用函数: {func.__name__}")
            result = func(*args, **kwargs)
            print(f"[{level}] 完成调用函数: {func.__name__}")
            return result

        return wrapper

    return decorator


# ========== 类装饰器 ==========


class DebugDecorator:
    """
    类形式的调试装饰器 - 功能与 debug 函数装饰器相同
    用法: @DebugDecorator

    优点: 可以保存状态，扩展性更好
    """

    def __init__(self, func: F) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.call_count = 0  # 可以记录调用次数等状态

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        print(f"[DEBUG] 开始调用函数: {self.func.__name__} (第{self.call_count}次)")
        result = self.func(*args, **kwargs)
        print(f"[DEBUG] 完成调用函数: {self.func.__name__}")
        return result


class LogDecorator:
    """
    带参数的类装饰器 - 支持自定义日志级别
    用法: @LogDecorator("INFO")

    这是一个装饰器工厂类，返回真正的装饰器函数
    """

    def __init__(self, level: str) -> None:
        self.level = level

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print(f"[{self.level}] 开始调用函数: {func.__name__}")
            result = func(*args, **kwargs)
            print(f"[{self.level}] 完成调用函数: {func.__name__}")
            return result

        return wrapper


# ========== 测试函数 ==========


@debug
def say_hi() -> None:
    """使用函数装饰器的示例"""
    print("Hi there!")


@log("INFO")
def say_hello() -> None:
    """使用带参数函数装饰器的示例"""
    print("Hello world!")


@DebugDecorator
def show_example() -> None:
    """使用类装饰器的示例"""
    print("This is an example")


@LogDecorator("WARNING")
def show_sample() -> None:
    """使用带参数类装饰器的示例"""
    print("This is a sample")


def plain_function() -> None:
    """普通函数，用于演示动态装饰"""
    print("I'm a plain function")


def main() -> None:
    """主函数 - 演示各种装饰器的使用方法"""

    print("=" * 50)
    print("装饰器使用示例")
    print("=" * 50)

    # 1. 使用装饰器语法糖
    print("\n1. 使用装饰器语法糖 (@decorator):")
    say_hi()
    say_hello()
    show_example()
    show_sample()

    # 2. 动态应用函数装饰器
    print("\n2. 动态应用函数装饰器:")
    decorated_func1 = debug(plain_function)
    decorated_func1()

    decorated_func2 = log("ERROR")(plain_function)
    decorated_func2()

    # 3. 动态应用类装饰器
    print("\n3. 动态应用类装饰器:")
    decorated_func3 = DebugDecorator(plain_function)
    decorated_func3()

    decorated_func4 = LogDecorator("CRITICAL")(plain_function)
    decorated_func4()

    # 4. 演示类装饰器的状态保持
    print("\n4. 类装饰器的状态保持:")
    show_example()  # 第二次调用，会显示调用次数

    print("\n装饰器演示完成！")


if __name__ == "__main__":
    main()
