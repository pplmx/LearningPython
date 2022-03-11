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


if __name__ == '__main__':
    s1 = Singleton()
    s2 = Singleton()
    s3 = Singleton()
    s1.shared_attr = 1
    s1.shared_func = lambda x: x * x
    assert s2.shared_attr == 1
    assert s2.shared_func(3) == 9

    print(s3.shared_attr)
    print(s3.shared_func(3))
