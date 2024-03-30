class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None

class DoubleLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if not self.head:
            self.head = Node(data)
        else:
            new_node = Node(data)
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
            new_node.prev = cur

    def find_max(self):
        if not self.head:
            return None
        else:
            max_val = self.head.data
            cur = self.head
            while cur:
                max_val = max(max_val, cur.data)
                cur = cur.next
            return max_val

# Create a double linked list
dll = DoubleLinkedList()

# Append some values
dll.append(5)
dll.append(10)
dll.append(150)
dll.append(20)
dll.append(25)

# Find maximum value
max_value = dll.find_max()

print("Maximum value in the double linked list is:", max_value)