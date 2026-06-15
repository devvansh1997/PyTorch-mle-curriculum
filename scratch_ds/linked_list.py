from __future__ import annotations


class Node:
    def __init__(self, data):
        self.data = data
        self.next: Node | None = None


class LinkedList:
    def __init__(self) -> None:
        self.head = None  # because we are starting with nothing

    def prepend(self, data):
        """
        add a node to the beginning of the linked list
        """
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def append(self, data):
        """add a node to the very end"""
        new_node = Node(data)

        # check if list is empty
        if not self.head:
            self.head = new_node
            return

        # if not - we need to find the last element in the linkedlist
        current = self.head
        while current.next:
            current = current.next

        # we are now at the last element
        current.next = new_node

    def display(self):
        """disply the entire list"""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements) + " -> None")
