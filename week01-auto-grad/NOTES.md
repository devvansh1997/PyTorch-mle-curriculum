# Week 1 — Tensor + Autograd Internals · Revision Notes

---

## 🚩 Flagged for later (end of Week 2)

- **LayerNorm backward derivation by hand.** Drill skipped on Day 3.
  Standard ML interview question. Need to derive `∂L/∂x_i` given upstream `∂L/∂y_i`.
  Hard part: `x_i` affects every `y_j` via three coupled paths (direct, through `μ`, through `σ²`).
  Final formula has the shape:
  `∂L/∂x_i = (1/(D·s)) · [D·(∂L/∂x̂_i) − Σ(∂L/∂x̂_j) − x̂_i · Σ(∂L/∂x̂_j · x̂_j)]`
  Revisit with paper + a numerical example.

---

## Part 1 — Tensor Puzzle Takeaways (Sasha Rush, #1–10)

### The mental model
Forget loops. Forget reshape tricks. Everything is:
- **`arange`** to generate index vectors
- **Broadcasting** via `[:, None]` / `[None, :]` to turn 1D into 2D
- **Comparison ops** to produce boolean masks
- **`where`** as a vectorized if/else
- **Matmul (`@`)** to do weighted sums in disguise

### Recurring patterns to remember

**1. Sum without `sum`**
```python
a @ ones(len(a))[:, None]
```
Matmul with a column of ones = sum. Same trick scales to weighted sums and cumulative sums.

**2. Cumulative sum via lower-triangular mask**
```python
(arange(n)[:, None] >= arange(n)[None, :]).long() @ a
```
The mask matrix has 1s on and below the diagonal. Multiplying by `a` accumulates.

**3. Outer product = column @ row**
```python
a[:, None] @ b[None, :]
```
Shape `(i, 1) @ (1, j)` → `(i, j)`. Every broadcasting puzzle uses this.

**4. Diagonal via fancy indexing**
```python
a[arange(n), arange(n)]
```
Pairs each row index with the same column index → pulls out the diagonal.

**5. Identity matrix without `eye`**
```python
(arange(j)[:, None] == arange(j)[None, :]).long()
```
Equality check between row and column indices = 1 on diagonal, 0 elsewhere.

**6. Upper triangular without `triu`**
```python
(arange(j)[:, None] <= arange(j)[None, :]).long()
```
Same trick but with `<=` instead of `==`.

**7. Stacking with `where`**
```python
where(arange(2)[:, None] == 0, a, b)
```
Row 0 picks from `a`, row 1 picks from `b`. The `[:, None]` makes the row index broadcast across columns.

**8. Cyclic shift via modulo indexing**
```python
a[(arange(i) + 1) % i]
```
Build the wrapped index array first, then index `a` with it once. Never index out of range.

### Rules of the game
1. Solve in **1 line, <80 cols**.
2. Allowed: `@`, arithmetic, comparison, `.shape`, any indexing, previous puzzles.
3. Banned: `view`, `sum`, `take`, `squeeze`, `tensor`.
4. Helpers: `arange`, `where`.

### Things I struggled with
- **roll**: tried to index first and mask second — crashed on out-of-range. Fix: **build the wrapped index array first, then index once.**
- **Forgetting `[:, None]`**: most failures came from shape mismatches I didn't visualize. Always write out shapes before coding.

---

## Part 2 — Anki Concepts (drill these)

### Q1: What is a tensor's **stride**?
**Stride** = number of elements to skip in memory to move 1 step along a given dimension.

- Shape `(3, 4)` row-major → stride `(4, 1)`
  - Move 1 row → skip 4 elements
  - Move 1 col → skip 1 element
- The data lives in **one flat 1D array**. Stride maps `(i, j)` → flat offset:
  `offset = i * stride[0] + j * stride[1]`
- **Transpose is free** — it just swaps strides, no data copied.

---

### Q2: What does **contiguous** mean?
A tensor is contiguous when its memory layout matches row-major (C-order) — rightmost dim changes fastest. Equivalently: strides equal `(prod of trailing dims, ..., 1)`.

**Operations that break contiguity (produce non-contiguous views):**
- `transpose` / `.T`
- `permute`
- Stepped slicing like `a[::2]`

**Fix:** `.contiguous()` — allocates and copies into a fresh contiguous buffer.

Basic slicing like `a[:, 1]` returns a contiguous 1D tensor — does NOT break it.

---

### Q3: Difference between `view` and `reshape`?

| | `view` | `reshape` |
|---|---|---|
| Requires contiguous? | Yes | No |
| Copies data? | Never | Only if needed |
| Fails on non-contiguous? | Yes (RuntimeError) | No (silently copies) |

**Rule of thumb:**
- `view` → use when you *know* tensor is contiguous. Fails loudly if assumption breaks (good — catches bugs).
- `reshape` → always works, but you don't see the copy.

```python
a = torch.zeros(3, 4)
b = a.T              # non-contiguous
b.view(12)           # RuntimeError
b.reshape(12)        # works, silently copies
```

---

### Q4: Broadcasting rules

Applied **right-to-left**, dim by dim:

1. **Right-align** the shapes (line up by last axis).
2. Pad the shorter shape with **1s on the left** until they have equal rank.
3. Two dims are compatible if **equal OR one is 1**.
4. Result shape takes the **max** along each dim.
5. Any dim that was 1 gets stretched — **no memory copy** (stride set to 0).

**Examples:**
```
(3, 4) and (4,):
  → (3, 4) vs (1, 4)  ✅  result (3, 4)

(3, 4) and (3,):
  → (3, 4) vs (1, 3)  ❌  4 vs 3, neither is 1

(5, 1, 4) and (3, 4):
  → (5, 1, 4) vs (1, 3, 4)  ✅  result (5, 3, 4)
```

**Why right-aligned?** So the innermost dims (features/channels) line up first, and outer dims (batch) get broadcast naturally.

---

## Part 3 — One-line revision cheat sheet

| Concept | Mnemonic |
|---|---|
| Stride | "Skips per step" |
| Contiguous | "Row-major in memory" |
| `view` | "Cheap, strict" |
| `reshape` | "Safe, sometimes copies" |
| Broadcasting | "Right-align, pad with 1s, match or one-is-1" |
| Puzzle trick | "Build index array → index once" |
| Puzzle trick | "`[:, None]` × `[None, :]` = outer-product shape" |
