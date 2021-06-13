#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Node class
from typing import Optional


class Node:

    # Function to initialize the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null

    def __repr__(self):
        return f'{self.data}'


# Linked List class contains a Node object
class LinkedList:

    # Function to initialize head
    def __init__(self):
        self.head = None
        self.len = 0

    def is_empty(self):
        return self.head is None

    def length(self):
        return self.len


class SinglyLinkedList(LinkedList):

    def __init__(self, to_linked: Optional[list] = None):
        """

        :param to_linked:
        """
        super().__init__()

        if to_linked:
            self.__transfer_list2singly_linked(to_linked)

    def __transfer_list2singly_linked(self, li):
        """
        [1, 3, 5]
        ==> transfer
        1 -> 3 -> 5
        :param li:
        :return:
        """
        for i in li:
            self.append(Node(i))

    def push(self, node: Node):
        """
        insert a node at the beginning
        :param node:
        :return:
        """
        self.len += 1
        node.next = self.head
        self.head = node

    def insert(self, idx: int, node: Node):
        """
        insert a node before the idx
        :param idx:
        :param node:
        :return:
        """
        if idx <= 0:
            self.push(node)
        elif idx > self.length() - 1:
            self.append(node)
        else:
            self.len += 1
            cur = self.head
            # index to the position
            while idx - 1:
                cur = cur.next
                idx -= 1
            node.next = cur.next
            cur.next = node

    def append(self, node: Node):
        """
        insert a node at the end
        :param node:
        :return:
        """
        self.len += 1
        if not self.head:
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = node

    def __iter__(self):
        cur = self.head
        while cur:
            # return a node
            yield cur
            cur = cur.next

    def __repr__(self):
        temp = self.head
        p_str = ''
        while temp:
            p_str += f'{temp.data} -> '
            temp = temp.next
        return p_str[:-4]


class DoublyLinkedList(LinkedList):
    ...


class CircularLinkedList(LinkedList):
    ...


def test_singly_linked_list_push():
    s_l_list = SinglyLinkedList()
    n1 = Node(1)
    n2 = Node(3)
    n3 = Node(5)
    n4 = Node(7)
    n5 = Node(9)
    s_l_list.push(n1)
    s_l_list.push(n2)
    s_l_list.push(n3)
    s_l_list.push(n4)
    s_l_list.push(n5)
    # 9 -> 7 -> 5 -> 3 -> 1
    print(s_l_list)


def test_singly_linked_list_append():
    s_l_list = SinglyLinkedList()
    n1 = Node(1)
    n2 = Node(3)
    n3 = Node(5)
    n4 = Node(7)
    n5 = Node(9)
    s_l_list.append(n1)
    s_l_list.append(n2)
    s_l_list.append(n3)
    s_l_list.append(n4)
    s_l_list.append(n5)
    # 1 -> 3 -> 5 -> 7 -> 9
    print(s_l_list)


def test_singly_linked_list_insert():
    # verify list to linked
    s_l_list = SinglyLinkedList([1, 3, 5, 7, 9])

    # verify <= 0
    s_l_list.insert(-1, Node(2333))
    # among the len
    s_l_list.insert(3, Node(55555))
    # verify > len
    s_l_list.insert(999, Node(6666))

    # verify whether it can be iterable
    for sl in s_l_list:
        print(sl)

    # verify the output whether it already was redefined
    print(s_l_list)
    print(s_l_list.length())


if __name__ == '__main__':
    test_singly_linked_list_push()
    test_singly_linked_list_append()
    test_singly_linked_list_insert()
