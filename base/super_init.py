class Animal:
    def __init__(self):
        self.color = "black"


class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.age = 1


class Engine:
    def __init__(self):
        super().__init__()
        self.performance = 80


class Skeleton:
    def __init__(self):
        super().__init__()
        self.shape = "Rectangle"


class Car(Engine, Skeleton):
    def __init__(self):
        super().__init__()

    def deliver(self):
        print(self.performance)
        print(self.shape)


if __name__ == "__main__":
    BMW = Car()
    BMW.deliver()
    ci = Cat()
    print(ci.color, ci.age)
