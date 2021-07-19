#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod


def fibonacci(idx):
    a, b = 1, 2
    while a <= idx + 1:
        yield a
        a, b = b, a + b


class Animal:

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @classmethod
    def where_is(cls):
        print("Where is the animal?")

    @staticmethod
    def is_alive():
        return True


class Dog(Animal):

    def __init__(self, name, gender='male'):
        self.__name = name
        self.__gender = gender

    def run(self):
        print(f'Dog, {self.__name} runs fast.')

    # @classmethod
    # def where_is(cls):
    #     print("The dog is under the tree.")


class Cat(Animal):

    def __init__(self, name, gender='female'):
        self.__name = name
        self.__gender = gender

    def run(self):
        print(f'Cat, {self.__name} runs fast.')

    # @classmethod
    # def where_is(cls):
    #     print("The cat is on the tree.")


class Bird(Animal):

    def __init__(self, name, gender='female'):
        self.__name = name
        self.__gender = gender

    def run(self):
        print(f'Bird, {self.__name} flies fast.')

    @classmethod
    def where_is(cls):
        print("The bird is in the sky.")

    @staticmethod
    def is_alive():
        return False


class Horse(Animal):

    def __init__(self, name, gender='female'):
        self.__name = name
        self.__gender = gender

    def run(self):
        print(f'Horse, {self.__name} runs fast.')

    @classmethod
    def where_is(cls):
        print("The horse is on the grassland.")

    @classmethod
    def is_alive(cls):
        return False

    def cc():
        return True


if __name__ == '__main__':
    spotty = Dog("Spotty")
    mimi = Cat("Mimi")
    bee = Bird("Bee")
    hoo = Horse("Hoo")
    print(type(Horse.cc), Horse.cc)
    print()
    print(type(spotty.run), spotty.run)
    print(type(mimi.run), mimi.run)
    print(type(bee.run), bee.run)
    print(type(hoo.run), hoo.run)
    print()
    print(type(spotty.is_alive), spotty.is_alive)
    print(type(mimi.is_alive), mimi.is_alive)
    print(type(bee.is_alive), bee.is_alive)
    print(type(hoo.is_alive), hoo.is_alive)
    print()
    print(type(spotty.where_is), spotty.where_is)
    print(type(mimi.where_is), mimi.where_is)
    print(type(bee.where_is), bee.where_is)
    print(type(hoo.where_is), hoo.where_is)
    print()
    print(type(Dog.where_is), Dog.where_is)
    print(type(Cat.where_is), Cat.where_is)
    print(type(Bird.where_is), Bird.where_is)
    print(type(Horse.where_is), Horse.where_is)
    print()
    print(type(fibonacci), fibonacci)
