"""
Stack
1. abstract data type that serves as a collection of elements
2. linear data structure
3. makes it possible to implement a stack as a singly linked list and a pointer to the top

## with two main principal operations
1. Push: adds an element to the collection
2. Pop: removes the most recently added element that was not yet removed / removes an item from the top of the stack

--> if the stack is full and does not contain enough space to accept an entity to be pushed,
----> stack is then considered to be in an overflow state

## additional operations
3. peek: give access to the top without modifying the stack
Stack's alternative name is LIFO

LIFO (last in, first out)

stack structure
//
maxsize, top, items

method

init, push, pop, peek

python's built in data structure list can be used as stack
===============================================================
Stack in Python can be implemented using the following ways:

list
Collections.deque
queue.LifoQueue
================================================================
reference by wikipedia
https://en.wikipedia.org/wiki/Stack_(abstract_data_type)
reference by geeksforgeeks
https://www.geeksforgeeks.org/stack-in-python/?ref=lbp

"""

# 1. stack
"""
if the stack grows bigger than the block of memory that currently holds it,
then Python needs to do some memory allocations
"""

stack = []

stack.append(1)
stack.append(2)
stack.append(3)

print(stack.pop())

"""
append => push => O(1)
pop() => pop => O(n)
pop() Last in First out 
"""


# 2. deque
"""
Deque is preferred over the list in the case where we need quicker append and pop operations 
from both the ends of the container

In deque
pop and append O(1) time

"""

from collections import deque

stack = deque()

stack.append(1)
stack.append(2)
stack.append(3)

print(stack)

print(stack.pop())
print(stack.pop())
print(stack.pop())

print(stack)

# LifoQueue
"""
Queue module has a LIFO queue, which is basically a stack
Data is inserted into Queue using the put() and get() takes data out from the queue

"""

from queue import LifoQueue

stack = LifoQueue(maxsize=3)

print(stack.qsize())

stack.put(1)
stack.put(2)
stack.put(3)

print(stack.full())
print(stack.qsize())

print(stack.get())
print(stack.get())
print(stack.get())

print(stack.empty())

"""
Implementation using singly linked list:
The linked list has two methods addHead(item) and removeHead(). 
two methods run in constant time -> O(1)

getSize()
isEmpty()
peek()
push(value)
pop()

"""

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:

    def __init__(self):
        self.head = Node("head")
        self.size = 0

    def __str__(self):


    def getsize(self):
        """
        get stack size
        :return: print stack size
        """
        return self.size

    def is_empty(self):
        """
        if stack is empty, return True
        :return: return bool
        """
        if self.size == 0:
            return True
        else:
            return False

    def peek(self):
        """
        get the top data of the stack without delete
        :return: top data
        """
        if self.is_empty():
            print("stack is empty!!!")
        else:
            return self.head.next.data

    def push(self, data):
        """
        insert item at the top of the stack
        :param data:
        :return:
        """
        node = Node(data)
        node.next = self.head.next
        self.head.next = node
        self.size += 1

        return

    def pop(self):
        """
        get item at the top of the stack
        :return:
        """

        if self.is_empty():
            print("Stack is empty")
        else:
            popitem = self.head.next
            self.head.next = self.head.next.next
            self.size -= 1
            return popitem

stack = Stack()
for i in range(1,11):
    stack.push(i)

for _ in range(1,6):
    popitem=stack.pop()
    print(popitem.data)