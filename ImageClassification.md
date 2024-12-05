#  Questionnaire

## 1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
   Performance and image data accuracy are the primary factors for resizing to a larger size on CPU then to a smaller size on the GPU.
   For performance reason, all images need to be in uniform size to create tensors. To make it as uniform, the image crop might introduce spurious empty zone and degrade the data.
   and compose all data augmentations in a batch. Uniform image size, batch data augmentation increases GPU performance.

## 2. If you are not familiar with regular expressions, find a regular expression tutorial and some problem sets, and complete them. Have a look on the book's website for suggestions.

## 3. What are the two ways in which data is most commonly provided for most deep learning datasets?
item_tfms - applies transformation on each picture individually in CPU.
batch_tfms - applies transformation on batch of images in GPU.

## 4. Look up the documentation for L and try using a few of the new methods that it adds.
```
from fastcore.foundation import L

# Create an L object from a list of numbers
t = L(range(12))

# Reverse the list
t.reverse()

# Access elements using advanced indexing
print(t[0])  # Output: 11
print(t[3, 5])  # Output: [8, 6]

def square(x):
    return x ** 2

def is_even(x):
    return x % 2 == 0

print(t)
print(t.sum())
print(t.shuffle())
print(t.itemgot())
squared = t.map(square)
print(squared)
lambda_squared = t.map(lambda x: x ** 2)
print(lambda_squared)
even_numbers = t.filter(is_even)
print(even_numbers)
print(t.filter(lambda x: x % 2 == 1))

#chain filter and map
print(t.filter(lambda x: x % 2 == 1).map(lambda x : x ** 2))

cycle_iter = t.cycle()

for _ in range(20):
    print(next(cycle_iter), end=' ')

    
# list of tuple
t1 = L([1, 2, 3])
t2 = L(['a','b','c'])
t3 = L([1, 2, 3]).zip(L(['a','b','c']))
print(t3)

data = L([(1,'a'),(2,'b'),(3,'c')])
print()
print(data.itemgot(0))
print(data.itemgot(1))
```

   
    
