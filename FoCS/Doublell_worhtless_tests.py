"""
Author: Dwayne Dilbeck
The file defines a Doubly linked list class and helper functions
"""

# pylint: disable=too-few-public-methods
class Node:
    """Node: Basic building block"""
    def __init__(self,data):
        self.val=data
        self.next=None
        self.prev=None

class Doublell:
    """Doubell: Made up of linked lists of Node"""
    def __init__(self):
        self.size=0
        self.head=None
        self.tail=None

    def ins_head(self,data):
        """ Insert new Node at Head"""
        new_node= Node(data)
        if self.head:
            self.head.prev = new_node
            new_node.next = self.head
        else:
            self.tail = new_node
        self.head = new_node
        self.size +=1

    def rm_head(self):
        """Remove Node from head"""
        temp_node = self.head
        self.head = self.head.next
        self.head.prev = None
        temp_node.next=None
        self.size -= 1
        return temp_node.val

    def ins_index(self,data,index):
        """Insert Node at Index"""
        curr = self.head
        # pylint: disable=unused-variable
        for i in range(0,index-1):
            curr = curr.next
        new_node = Node(data)
        new_node.prev = curr
        new_node.next = curr.next
        curr.next = new_node
        new_node.next.prev = new_node
        self.size+=1

    def rm_index(self,index):
        """Remove Node at the index"""
        curr = self.head
        # pylint: disable=unused-variable
        for i in range(0,index-1):
            curr = curr.next
        temp = curr
        curr.next.prev = curr.prev
        curr.prev.next = curr.next
        temp.next = None
        temp.prev = None
        self.size -= 1
        return temp.val

    def rm_tail(self):
        """Remove Node at the Tail"""
        temp_node = self.tail
        self.tail = self.tail.prev
        temp_node.prev=None
        self.size -= 1
        return temp_node.val

    def ins_tail(self,data):
        """Insert a new Node at the Tail"""
        new_node = Node(data)
        if self.tail:
            self.tail.next = new_node
            new_node.prev = self.tail
        else:
            self.head = new_node
        self.tail = new_node
        self.size +=1

    def to_string(self):
        """Create a string from the linked list"""
        if self.head:
            curr = self.head
            val = str(self.head.val)
            while curr.next:
                curr = curr.next
                val += str(curr.val)
            return val
        return None

    def to_rev_string(self):
        """Create a string by transversing in reverse"""
        if self.tail:
            curr = self.tail
            val = str(curr.val)
            while curr.prev:
                curr = curr.prev
                val += str(curr.val)
            return val
        return None


if __name__ == "__main__":
    dataset = Doublell()
    print("String:"+str(dataset.to_string()) )
    print("rev:"+str(dataset.to_rev_string()))
    dataset.ins_tail("1")
    dataset.ins_tail("2")
    dataset.ins_tail("3")
    dataset.ins_tail("4")
    print(dataset.to_string())
    print(dataset.to_rev_string())
    print(dataset.rm_head())
    print(dataset.to_string())
    dataset.ins_index("9",2)
    print(dataset.to_string())
    print(dataset.to_rev_string())
    print(dataset.rm_index(2))
    print(dataset.rm_tail())
    print(dataset.to_string())
    print(dataset.to_rev_string())
