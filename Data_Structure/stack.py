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