# kodiak — offline LeetCode-style practice

Solutions live at the top level (e.g. `number_of_islands.py`). Each one ends with:

```python
if __name__ == "__main__":
    from _harness.runner import run_tests
    run_tests("number_of_islands", Solution().numIslands)
```

Run a solution against its test cases:

```bash
cd kodiak && conda run -n mle-curriculum python number_of_islands.py
```

Test cases are JSON files under `_harness/test_cases/<problem_name>.json`. Each case is `{"name": ..., "args": [...], "expected": ...}`, where `args` is splatted positionally into the solution function.
