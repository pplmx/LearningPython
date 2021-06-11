#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Node class
class Node:

    # Function to initialize the node object
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null


# Linked List class contains a Node object
class LinkedList:

    # Function to initialize head
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def length(self):
        temp = self.head
        idx = 0
        while temp:
            idx += 1
            temp = temp.next
        return idx

    def __repr__(self):
        temp = self.head
        p_str = ''
        while temp:
            p_str += f'{temp.data} -> '
            temp = temp.next
        return p_str[:-4]


class SinglyLinkedList(LinkedList):
    ...


class DoublyLinkedList(LinkedList):
    ...


class CircularLinkedList(LinkedList):
    ...


def test_singly_linked_list():
    n1 = Node(1)
    n2 = Node(3)
    n3 = Node(5)
    n4 = Node(7)
    n5 = Node(9)
    s_l_list = SinglyLinkedList()
    s_l_list.head = Node(0)
    s_l_list.head.next = n1
    n1.next = n2
    n2.next = n3
    n3.next = n4
    n4.next = n5

    print(s_l_list)
    print(s_l_list.length())


if __name__ == '__main__':
    test_singly_linked_list()
