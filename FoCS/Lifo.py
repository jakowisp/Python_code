"""
Author: Dwayne Dilbeck
Implement a last in first out data structure, typically a 'Stack'
"""

from DoubleLinkedList import DoubleLinkedList


class Lifo:
    """The Last in First out class Lifo"""
    def __init__(self):
        self.linked_list = DoubleLinkedList()

    def put(self,data):
        """ Add data at Head"""
        self.linked_list.ins_head(data)

    def get(self):
        """Removes data from head"""
        return self.linked_list.rm_head()

if __name__ == "__main__":
    stack = Lifo()
    stack.put("a")
    stack.put("b")
    stack.put("c")
    stack.put("d")
    stack.put("e")
    assert stack.get() == "e", "lifo is bad"
