from collections import deque

def bfs_reachable(grid: list[list[int]], start_r: int, start_c: int) -> set[tuple[int, int]]:
    """
    grid:    2D list, 0 = free, 1 = obstacle
    start:   ego's (row, col) position
    returns: set of (r, c) for every free cell reachable from start
    """
    H, W = len(grid), len(grid[0])

    # edge case: ego is sitting on an obstacle — nothing reachable
    if grid[start_r][start_c] == 1:
        return set()

    visited = {(start_r, start_c)}
    queue = deque([(start_r, start_c)])

    # 4-connected neighbor offsets: down, up, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # bounds check FIRST, then grid access
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return visited

grid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]

start = (0, 0)

result = bfs_reachable(grid, *start)
assert len(result) == 10
assert (0, 3) not in result      # right half unreachable
assert (4, 1) in result          # bottom of left half reachable