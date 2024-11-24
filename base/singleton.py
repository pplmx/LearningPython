class Singleton:
    # state shared by each instance
    __shared_state = dict()

    # constructor method
    def __init__(self):
        self.__dict__ = self.__shared_state


class Single:
    __shared_state = {}

    def __new__(cls, *args, **kwargs):
        obj = super(Single, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls.__shared_state
        return obj


class SingletonMeta(type):
    _instances = {}  # 存储类的唯一实例

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SingletonClass(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value

    def display_value(self):
        print(f"The value is: {self.value}")


if __name__ == "__main__":
    s1 = Singleton()
    s2 = Singleton()
    s3 = Singleton()
    s1.shared_attr = 1
    s1.shared_func = lambda x: x * x
    assert s2.shared_attr == 1
    assert s2.shared_func(3) == 9

    print(s3.shared_attr)
    print(s3.shared_func(3))
    print(s1 is s2)  # False, 这是 Borg's Singleton, 每个实例都是新的, 只是状态是共享的
    print(type(Singleton))  # <class 'type'>

    instance1 = SingletonClass(10)
    instance2 = SingletonClass(20)

    instance1.display_value()  # 输出: The value is: 10
    instance2.display_value()  # 也输出: The value is: 10

    print(instance1 is instance2)  # 输出: True
    print(type(SingletonClass))  # <class '__main__.SingletonMeta'>
