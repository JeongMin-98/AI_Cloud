
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
over a conventional array is that the list elements can be easily inserted or removed without reallocation or reorganization
of the entire structure becaurse the data items need not be stored contiguously in memory or on disk

### Disadvantages
Acess time is linear time and difficult to pipeline
Arrays have better cache locality compared to linked lists.
do not allow random access to the data or any form of efficient indexing, many basic operations.


Linked lists are among the simplest and most common data structures.
can be used to implement several other common abstract data types, including lists, statcks, queues, associative arrays,
and S-experssions.


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









