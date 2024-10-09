## Questionnaire
#  1. How is a grayscale image represented on a computer?
In computer, everything is number. It represents the grayscale image as matrix of numbers. 0 represents white, 255 represents black color. Range of other number between (0, 255) represents grey shade.

#  2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
The MNIST dataset follows a common layout. It contains seperate training set and validation set. Inside training set, there 3 and 7 folders contain images. 3 and 7 are labels or target of datasets.
Seperating the training and validation set produces the good model behavior to prevent the overfit or underfit.

#  3. Explain how the "pixel similarity" approach to classifying digits works
1. Calculate the average pixel of 3 digit images 2. Calculate the average pixel of 7 digit images 3. find the average pixel of the given image 4. find closest average pixel value of 3s and 7s images

#  4. What is a list comprehension? Create one now that selects odd numbers from list and doubles them.
List comprehension produces the list data structure filled with numbers from enumerating the range of numbers in loop construct. Conditional filter is applicable.
```
list = [item*2 for item in range(1,10) if item % 2]
>>> list
[2, 6, 10, 14, 18]
```
#  5. What is a rank-3 tensor?
rank-3 tensor contains data in 3 dimensions. width, length, height or depth. For example, book which has 3 dimensions where numbers of page is depth or height, each page has fixed width and length.

# 6. What is the difference between tensor rank and shape? How do you get the rank from shape?
```
import torch

#construct tensor
t = torch.rand(9, 7, 5)

# shape of tensor says number of elements in all dimensions
print(f"The shape of the tensor is: {t.shape}")
print(f"The size of the tensor is: {t.size()}")

# print the size of the second dimension
print(f"The size of the second dimension is: {t.size(1)}")

# rank of tensor says number of dimensions in tensor
print(f"The rank of the tensor is: {t.dim()}")
print(f"The rank of the tensor is: {t.ndim}")
```

# 7. What are RMSE and L1 norm?
RMSE means root mean squared error or L2 norm. 1) calculate the absolute value of tensors say absdiff 2) square of absdiff say sqrofabsdiff 3) take sqaure root of sqrofabsdiff. L1 norm means absolute value of differences.

```
#L1 norm. a and b tensors
L1 = (a - b).abs().mean()  
L2 = ((a - b)**2).mean().sqrt()
```

# 8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
Using NumPy arrays, PyTorch tensors, we can accomplish thousands of numerical operations quickly. Because these data structures are memory compact structure, lives entirely in GPUs and numerical operations are implemented in optimized C code.

# 9. Create a 3X3 tensor or array containing the numbers from 0 to 9. Double it. Select the bottom-right four numbers
```
import torch

data = [[1,2,3],[4,5,6],[7,8,9]]
tns = torch.tensor(data)

tns = tns * 2
tns

tns[2:]
```
# 10. What is broadcasting?
When tensor operation applied on two different rank tensors, lower rank tensor automatically expands to higher rank tensor shape and will do elementwise operations. This is known as broadcasting. It eases the expressivity of mathematical operation and efficiency.
```
tensor([1,2,3]) + tensor([1])
tensor([2,3,4])
```

# 11. Are metrics generally calculated using the training set or validation set? why?
Metrics are calculated using validation set. Becuase it prevents over-fitting the model by validating the new data set which was not present in training set.

# 12. What is SGD?
Stochastic Gradient Descent is an optimization technique to maximize the performance by assigning the weights

# 13. Why does SGD use mini-batches?
SGD uses few data items to calculate the average loss in each epoch. It is called mini-batches. It helps to train the model quickly and accurately. Using whole data set will provide stable, accurate model but it takes long time to train it. On other hand, considering a single item for training brings imprecise and unstable model. Thus, choosing the correct batch size is one of the key decision in deep learning. In addition, GPU would not be performant enough if it uses either single item or whole data set.

# 14. What are the seven steps in SGD for machine learning?
Step 1 : Initialize the parameters
Step 2 : Calculate the predictions
Step 3 : Calculate the loss
Step 4 : Calculate the gradients
Step 5 : Step the weights
Step 6 : Repeat the process
Step 7 : stop on minimal loss

# 15. How do we initialize the weights in a model?
We initialize the parameters to random values. For example, linear model contains three parameters a,b,c. Using PyTorch, we are initializing the parameters and track their gradient as follows
```
params = torch.randn(3).requires_grad_()
```

# 16.What is loss?


