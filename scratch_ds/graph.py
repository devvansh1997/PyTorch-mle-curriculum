from collections import deque


class Graph:
    def __init__(self):
        """creating an empty node with no neighbors"""
        self.adj_list = {}

    def add_vertex(self, vertex):
        """add a new node to the graph"""
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, v1, v2):
        """create a bidirectional connection between two vertex"""
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.adj_list[v1].append(v2)
        self.adj_list[v2].append(v1)

    def bfs(self, start_vertex):
        """perform a bread first search"""
        queue = deque([start_vertex])  # our FIFO queue
        visited = {start_vertex}  # our VISITED set
        result = []  # final result of traversal

        while queue:
            # pop the oldest (first one in the list) element in the queue
            current = queue.popleft()
            result.append(current)

            # now we need to look at all the neighbors of our current
            for neighbor in self.adj_list[current]:
                # we only want to traverse a neighbor if they HAVE NOT BEEN VISITED
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start_vertex, visited=None, result=None):
        """perform depth first traversal"""
        # if we are starting out - absolute first call on the graph
        # then we need to instantiate the visited and results variables
        if visited is None:
            visited = set()
        if result is None:
            result = []

        # now we add the current node to visited list (Since we are sitting on it)
        # and add to the result array
        visited.add(start_vertex)
        result.append(start_vertex)

        for neighbor in self.adj_list[start_vertex]:
            if neighbor not in visited:
                self.dfs(neighbor, visited, result)

        return result
