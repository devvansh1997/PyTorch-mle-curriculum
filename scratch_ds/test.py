from graph import Graph

# Create the graph
g = Graph()
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "D")
g.add_edge("B", "E")
g.add_edge("C", "F")

print("BFS (Level-by-Level):", g.bfs("A"))
print("\n\n")
print("DFS (Level-by-Level):", g.dfs("A"))
