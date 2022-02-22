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