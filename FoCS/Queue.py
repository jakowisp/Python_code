"""
Author:DWayne Dilbeck
Implement a queue with linked lists
"""
from DoubleLinkedList import DoubleLinkedList

# pylint: disable=invalid-name
class Queue:
    """Implement the Queue class"""
    def __init__(self):
        self.linked_list = DoubleLinkedList()

    def put(self,data):
        """ Insert at the tail"""
        self.linked_list.ins_tail(data)

    def get(self):
        """Remove data at head"""
        return self.linked_list.rm_head()

if __name__ == "__main__":
    stack = Queue()
    stack.put("Q")
    stack.put("u")
    stack.put("e")
    stack.put("u")
    stack.put("e")
    assert stack.get()=="Q","Queue returned wrong data"
