
"""
## LinkedList
1. A linear collection of data elements whose order is not given by their physical placement in memory
Instead each elememt points to the next.
2. data structure consisting of a collection of nodes which together represent a sequence.


### Basic Form
each noed contains: data and a reference ( in other words, a link) to the next node in sequence

        ===============================                  ===============================
        =    data    =    reference   = ==============>  =    data    =    reference   =
        ===============================                  ===============================

### Advantages
allows for efficient insertion or removal of elements from any position in the sequence during iterations
over a conventional array is that the list elements can be easily inserted or removed without reallocation or
reorganization of the entire structure because the data items need not be stored contiguously in memory or on disk

### Disadvantages
Access time is linear time and difficult to pipeline
Arrays have better cache locality compared to linked lists.
do not allow random access to the data or any form of efficient indexing, many basic operations.


Linked lists are among the simplest and most common data structures.
can be used to implement several other common abstract data types, including lists, stacks, queues, associative arrays,
and S-expressions.

reference by wiki
https://en.wikipedia.org/wiki/Linked_list
"""


class Node:

    def __init__(self, data, next=None):
        self.data = data
        self.next = next

class SingleLinkedList:

    def __init__(self):
        self.head = None

    def linked_list_print(self):

        if self.head is None:
            print("empty Linked list")
        else:
            temp = self.head
            while temp is not None:
                print(temp.data)
                temp = temp.next

    def add_first(self, node):

        if self.head is None:
            self.head = Node

        else:
            temp = self.head
            self.head = node
            self.head.next = temp

    def add_last(self, node):
        """
        Linked list add node at last.
        :param node:
        :return:
        """

        temp = self.head
        while temp.next is not None:
            temp = temp.next

        temp.next = node

    def delete_node(self, key):

        if self.head.next is None:
            print("empty Linked List")
            return

        temp = self.head

        if temp.data == key:
            self.head = temp.next
            del temp
            return

        while temp.data != key:
            before = temp
            temp = temp.next

        if temp.next is None:
            del temp
            before.next = None

        else:
            before.next = temp.next
            del temp







e1 = Node("hello")
e2 = Node("world")
e3 = Node("add first")
e4 = Node("add Last")

LList = SingleLinkedList()

LList.head = e1
LList.head.next = e2
LList.linked_list_print()

LList.add_first(e3)
LList.linked_list_print()

LList.add_last(e4)
LList.linked_list_print()

LList.delete_node("add first")
LList.linked_list_print()











