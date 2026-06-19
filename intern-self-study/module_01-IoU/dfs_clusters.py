def count_clusters(grid: list[list[int]]) -> int:
    """
    grid: 2D List that represents a grid where 0 = free & 1 = obstacle
    returns: number of connected obstacle clusters (4 - connected)
    """

    if not grid or not grid[0]:
        return 0
    
    # get boudaries
    H, W = len(grid), len(grid[0])
    visited = set()
    count = 0

    def dfs(r: int, c: int) -> None:
        # base case: out of bounds, not obstacle, or already visited
        if not (0 <= r < H and 0 <= c < W):
            return 
        if grid[r][c] != 1 or (r,c) in visited:
            return
        
        # good node - add to visited
        visited.add((r,c))
        dfs(r - 1, c)
        dfs(r + 1, c)
        dfs(r, c - 1)
        dfs(r, c + 1)

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1 and (r,c) not in visited:
                count += 1
                dfs(r,c)
    
    return count

grid = [
    [1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
]

print(count_clusters(grid))