#  Questionnaire

## 1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
   Performance and data quality are the primary factors for resizing to a larger size on CPU then to a smaller size on the GPU.
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
numbers = L(range(12))

# Reverse the list
numbers.reverse()


# square function
def square(x):
    return x ** 2

# checks whether even number
def is_even(x):
    return x % 2 == 0

# Access elements using advanced indexing
print("Accessing number by its index :",numbers[0])  # Output: 11
print("Accessing number by its range :",numbers[3, 5])  # Output: [8, 6]
print("The sum of numbers :",numbers.sum())
print("Before shuffle :", numbers)
print("Shuffling the numbers :", numbers.shuffle()) #changes ordering

#map example
print("map example : transforms numbers to its square")
print(numbers)
squared = numbers.map(square) #transforms numbers to its square value
print("After applying square :", squared)
lambda_squared = numbers.map(lambda x: x ** 2) #transforms numbers using lambda
print("Using lambda :",lambda_squared)

# filter example
print("filter example : selects number based on condition")
even_numbers = numbers.filter(is_even)
print("Even numbers :", even_numbers)
print("Odd numbers using lambda :",numbers.filter(lambda x: x % 2 == 1))

# chaining map and filter example
print("chaining of map and filter :", numbers.filter(lambda x: x % 2 == 1).map(lambda x : x ** 2))

# creating cyclic loop
cycle_iter = numbers.cycle()
print("Cyclic sequence :", end=' ')
for _ in range(20):
    print(next(cycle_iter), end=' ')

#Append newline
print()

data = L([(1,'a'),(2,'b'),(3,'c')])
print("List of tuples :", data)
print("Accessing the first element of tuples : ", data.itemgot(0))
print("Accessing the second element of tuples :", data.itemgot(1))

#output
Accessing number by its index : 11
Accessing number by its range : [8, 6]
The sum of numbers : 66
Before shuffle : [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
Shuffling the numbers : [8, 7, 4, 10, 9, 11, 3, 1, 6, 2, 0, 5]
map example : transforms numbers to its square
[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
After applying square : [121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0]
Using lambda : [121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0]
filter example : selects number based on condition
Even numbers : [10, 8, 6, 4, 2, 0]
Odd numbers using lambda : [11, 9, 7, 5, 3, 1]
chaining of map and filter : [121, 81, 49, 25, 9, 1]
Cyclic sequence : 11 10 9 8 7 6 5 4 3 2 1 0 11 10 9 8 7 6 5 4 
List of tuples : [(1, 'a'), (2, 'b'), (3, 'c')]
Accessing the first element of tuples :  [1, 2, 3]
Accessing the second element of tuples : ['a', 'b', 'c']
```

## 5. Look up the documentation for the Python pathlib module and try using a few methods of the Path class.
```
from pathlib import Path

# Create a Path object for current dir
path = Path('.')
#Create a Path object for file
file_path = Path('/notebooks/Sample.ipynb')

#List dirs under current dir
list_subdirs = [item for item in path.iterdir() if item.is_dir()]
print(list_subdirs)

#Navigating file
notebookfile_path = path / file_path /'Sample.ipynb'

python_files = list(path.glob('**/*.ipynb'))
python_files

# Querying Path properties
print("Is file exist :", file_path.exists())

# Read the contents of a file
with file_path.open('r') as file_path:
    content = file_path.read()
print(len(content))

# Write to a file
new_file_path = path / 'hello.txt'
with new_file_path.open('w') as file:
    file.write('Hello, World!')

print("Is new_file_path exist :", new_file_path.exists())
```

## 6. Give two examples of ways that image transformations can degrade the quality of the data.
1. Rotating the image 45 degrees fills the corner regions of the new bounds with emptiness
2. Image rotation ad zoom interpolates the pixed which lowers the quality

## 7. What method does fastai provide to view the data in a DataLoaders?
data loader's show_batch method provides the image view from data loaders. It takes nrow and ncolumn as integer parameters.

## 8. What method does fastai provide to help you debug a Datablock?
data loader summary method provides a lot of details to debug the input 

## 9. Should you hold off on training a model until you have thoroughly cleaned your data?
No, we can start training the model to establish the baseline results on clean data.

## 10. What are the two pieces that are combined into cross-entropy loss in PyTorch?
1. softmax 2. Log liklihood 

## 11. What are the two properties of activations that softmax ensures? Why is this important?
The softmax ensures that all activations are all between 0 and 1 and they sum to 1. These two properties normalize the activations so we can choose the activation which is higher value over n cateogory and we do not want to worry about their relative higher or lower or both.

## 12. When might you want your activations to not have these two properties?
  In case binary classification problems, we do not want activations to follow these properties. Higher activation belongs to one category otherwise it does belong to other category. 

## 13. Calculate thh exp and softmax columns of figure 5-3 yourself.
|Activation|	Exp	|Softmax|
|----------|---------|-------|
|0.02|	1.02020134	|0.222098944|
|-2.49|	0.082909967	|0.01804959|
|1.25	|3.490342957	|0.759851466|
|SUM	|4.593454264	|1|


## 14. Why can't use torch.where to create a loss function for datasets where our label can have more than two categories?
torch.where works based boolean condition selects the items. Based on boolean condition, this tranforms the data set into two categories so in multi-category classification torch.where is not applicable.

## 15. What is the value of log(-2)? why?
Natural logarithm of any value is undefined. Lets assume log base e(-2) = x. e^x = -2. There is no value of x in exponent gives negative value so natural logarithm of negative value is undefined.

## 16.

## 16. What are good rules of thumb for picking a learning rate from the learning rate finder?


