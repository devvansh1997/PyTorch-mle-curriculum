from collections import deque

from _harness.runner import run_tests


class Solution:
    def shortest_binary_path(self, grid: list[list[int]]) -> int:
        # check impossible solution
        if grid[0][0] == 1 or grid[-1][-1] == 1:
            return -1

        # define rows and cols
        ROWS, COLS = (
            len(grid),
            len(grid[0]),
        )  # good for generalization - if matrix size changes

        # create queue as we will run BFS
        queue = deque([(0, 0)])
        distance = 1

        # mark the starting point as visited
        grid[0][0] = 1

        # define directions - 8 way
        directions = [
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
        ]

        # perform BFS now
        while queue:
            # keep track of each level
            for _ in range(len(queue)):
                row, col = queue.popleft()
                # base case
                if (row, col) == (ROWS - 1, COLS - 1):
                    return distance
                for delta_row, delta_col in directions:
                    new_row, new_col = row + delta_row, col + delta_col
                    if (
                        (0 <= new_row < ROWS)
                        and (0 <= new_col < COLS)
                        and (grid[new_row][new_col] == 0)
                    ):
                        grid[new_row][new_col] = 1  # updated to seen
                        queue.append((new_row, new_col))
            distance += 1
        return -1


if __name__ == "__main__":
    run_tests("shortest_path_binary_matrix", Solution().shortest_binary_path)
