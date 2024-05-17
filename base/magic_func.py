#!/usr/bin/env python


class A:
    # test __init__, __call__, __new__
    def __init__(self, *args, **kwargs):
        print(f"A.__init__ args: {args} kwargs: {kwargs}")
        print(f"A.__init__: {self}")
        super(A, self).__init__()

    def __call__(self):
        print("A.__call__")

    def __new__(cls, *args, **kwargs):
        print(f"A.__new__ args: {args} kwargs: {kwargs}")
        self = super(A, cls).__new__(cls)
        print(f"A.__new__: {self}")
        return self


def test_a():
    a = A(1, 2, 3, {"name": "jone"}, a=1, b=2)
    print("CanCallable: ", callable(a))
    a()


class B:
    # test __str__, __repr__
    def __str__(self):
        print("B.__str__")
        return "This is __str__"

    def __repr__(self):
        print("B.__repr__")
        return "This is __repr__"


class BB:
    # test __repr__

    def __repr__(self):
        print("BB.__repr__")
        return "This is __repr__"


class BBB:
    # test __str__

    def __str__(self):
        print("BBB.__str__")
        return "This is __str__"


def test_b():
    b = B()
    print(b)
    print(str(b))
    print(repr(b))
    print()
    bb = BB()
    print(bb)
    print(str(bb))
    print(repr(bb))
    print()
    bbb = BBB()
    print(bbb)
    print(str(bbb))
    print(repr(bbb))


class C:
    def __init__(self):
        print("C.__init__")
        self.test = "test"

    # 如果实例想在__init__之外, 声明新的实例属性, 则必须重写__getattr__方法
    def __getattr__(self, item):
        print(f"C.__getattr__ item: {item}")
        return self.__dict__.get(item, None)

    # # 实例属性赋值, 会调用__setattr__方法
    # def __setattr__(self, key, value):
    #     print(f'C.__setattr__ key: {key} value: {value}')
    #     super(C, self).__setattr__(key, value)
    #
    # # del 实例属性, 会调用__delattr__方法
    # def __delattr__(self, item):
    #     print(f'C.__delattr__ item: {item}')
    #     super(C, self).__delattr__(item)

    def __getattribute__(self, item):
        print(f"C.__getattribute__ item: {item}")
        return super().__getattribute__(item)


def test_c():
    c = C()
    print(c.test)
    print(c.a)
    c.a = "This is a"
    print(c.a)


if __name__ == "__main__":
    test_c()
