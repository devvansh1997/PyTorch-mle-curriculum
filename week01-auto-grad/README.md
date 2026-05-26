# week - 01 notes

[micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
- value class of the micrograd engine was fun to code
- learnt basics of a forward pass using handwritten class functions

now reading [internals of pytorch by edward yang](https://blog.ezyang.com/2019/05/pytorch-internals/)
- understood what a tensor is - the basic fundamental building block of the pytorch library
- tensors contains data and metadata about the tensor, typically they are - the size, stride, dtype, device, layout
- **stride** is something used to create a mapping for how and where is the data actually stored
- "TensorAccessor" class is super useful as it does the actual mapping for you - the indexing calc
- torch.mm is a 2D mat multiplication function - you get a dynamic dispatch which determines how to handle the dispatch based on the dtype, stride (since CPU vs CUDA are two different things compeletely - different libraries are used). the second dispatch is for the dtype.
- tensor extensions: there are a lot of tensor types so you need wrappers and these wrappers will do most of the heavy lifting. to understand a tensor we need 3 main things i.e. device, layout and dtype
- autograd engine is what makes pyorch so unique - its because of this automatic diffrentiation that we can abstract a lot of the difficult part out.. 

### before moving on to sasha's puzzles i need to know 3 concepts:
1. broadcasting - to be used when shapes are mismatched - and we want to perform some operation on the both matrices. we always do it along size-1 dimensions
```python
a = torch.tensor([1, 2, 3])          # shape (3,)
b = torch.tensor([[10], [20]])        # shape (2, 1)
a + b
# shape (2, 3):
# [[11, 12, 13],
#  [21, 22, 23]]
```
2. adding dimensions with ```None```. it is used for setting up broadcasting
```python
a = torch.arange(3) # shape (3,)
a[:,None] # shape (3,1) - column added
a[None, :] # shape (1,3) - row added
a[:,None] + a[None, :] # shape (3,3) - outer sum
```
3. ```torch.arange()``` is used for creating indices
```python
torch.arange(5) # tensor([0, 1, 2, 3, 4])
```
4. Boolean operations are replacing if/else statements
```<```, ```>```, ```==``` produce True/False tensors. You can multiply by them or sum them:
```python
a = torch.arange(5)
(a < 3)                    # [True, True, True, False, False]
(a < 3) * a                # [0, 1, 2, 0, 0]    ← zeros out where False
(a < 3).sum()              # 3                  ← counts True
```