import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"

CASES_DIR = Path(__file__).parent / "test_cases"


def _is_list_of_lists_of_numbers(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, list):
            return False
        for x in row:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                return False
    return True


def _normalize(value: Any) -> Any:
    if _is_list_of_lists_of_numbers(value):
        return sorted([list(row) for row in value])
    if isinstance(value, list):
        try:
            return sorted(value)
        except TypeError:
            return value
    return value


def run_tests(problem_name: str, solution_fn: Callable) -> None:
    cases_path = CASES_DIR / f"{problem_name}.json"
    with cases_path.open() as f:
        cases = json.load(f)

    print(problem_name)
    passed = 0
    name_width = max((len(c["name"]) for c in cases), default=20) + 2

    for case in cases:
        name = case["name"]
        args = case["args"]
        expected = case["expected"]

        start = time.perf_counter()
        try:
            result = solution_fn(*args)
            elapsed_ms = (time.perf_counter() - start) * 1000
            ok = _normalize(result) == _normalize(expected)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            ok = False
            result = f"<exception: {type(exc).__name__}: {exc}>"

        timing = f"{DIM}({elapsed_ms:.2f}ms){RESET}"
        if ok:
            print(f"  {GREEN}✓{RESET} {name.ljust(name_width)} {timing}")
            passed += 1
        else:
            print(
                f"  {RED}✗{RESET} {name.ljust(name_width)} "
                f"expected {expected!r}, got {result!r}  {timing}"
            )

    total = len(cases)
    print("  " + "─" * 36)
    color = GREEN if passed == total else RED
    print(f"  {color}{passed}/{total} passed{RESET}")

    sys.exit(0 if passed == total else 1)
