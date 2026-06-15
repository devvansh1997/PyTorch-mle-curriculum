from _harness.runner import run_tests


class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        """create DFS traversal to identify how many islands exist in a grid like structure"""
        if not grid or not grid[0]:
            return 0

        islands = 0
        ROWS, COLS = len(grid), len(grid[0])

        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        def dfs(r, c):
            # stack
            stack = [(r, c)]

            while stack:
                # get the current r,c by popping
                row, col = stack.pop()

                # check base cases
                if not (0 <= row < ROWS and 0 <= col < COLS):
                    continue
                if grid[row][col] != "1":
                    continue

                # if we are still here - then we have node to run 4-way DFS on
                # sink island
                grid[row][col] = "0"

                for delta_row, delta_col in directions:
                    stack.append((row + delta_row, col + delta_col))

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == "1":
                    islands += 1
                    dfs(r, c)

        return islands


if __name__ == "__main__":
    run_tests("number_of_islands", Solution().numIslands)
