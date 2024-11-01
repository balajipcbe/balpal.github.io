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
Loss is a metric describes how good model performs. Usually lower loss value indicates that model is performing good for their parameters assignments.
Higher loss values means model is bad for their parameters assignments.

# 17. Why can't we always use a high learning rate?
Choosing a high learning rate leads to large loss or diverge from progressive path. On other hand, choosing a lower learning rates takes more steps to reach the goal.

# 18. What is a gradient?
A gradient is a change of weight assignment on model parameters to make model parameters. In other words, When changing the model parameters up or down, leads to higher or lower the model loss.

# 19. Do you need to know how to calculate gradients yourself?
No, understanding derivative is good enough yet we do not need to calculate the gradient manually. Gradient calculating library is already implemented in PyTorch.

# 20. Why can't we use accuracy as a loss function ?
Accuracy represents the absolute value of model at specific parameters. But loss function is expected to be providing the change (gradient) of model at any parameters. After certain weight assignment, acuracy does NOT improve which does not allow the model to learn new. Mathematically, accuracy becomes almost constant thus gradient (change) becomes zero.

# 21. Draw the sigmoid function. What is special about its shape?
sigmoid function takes any input value and smooshes it into output value between 0 and 1. As a loss function, it provides higher confidence when predictions are not between 0 and 1.

# 22. What is the difference between a loss function and a metric?
A loss function should be a derivative which provide a change for each weight adjustment. This enables the model to learn for a small changes in weights instead of big flat or high steep. A metric is absolute value to indicate how model is doing on each epoch.

# 23. What is the function to calculate new weights using a learning rate?
train_epoch calculates the gradient and assigns new weight based on learning rate.

# 24. What does the DataLoader class do?
DataLoader class creates mini batches of dataset from any Python collection.

# 25. Write pseudocode showing the basic steps taken in each epoch for SGD.
Input : model, learning rate, parameters
step 1: iterate the data set as independent xb and dependent variable yb
step 1.1: calculate the gradient for given xb, yb, model
step 1.2: subtract the parameters weight based on parameter gradient and learning rate
step 1.3: update parameter gradient as zero

# 26. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1,a),(2,b),(3,c),(4,d)].What is special about that output data structure?
```
ds = L(enumerate([1,2,3,4], 'abcd'))
```
# 27. What does view do in PyTorch?
In PyTorch, View method changes the shape of tensor without changing its contents. parameter -1 makes the axis big to fit all the data.

# 28. What are the bias parameters in a neural network? Why do we need them?

# 29. What does the @ operator do in Python?
In Python, matrix multiplication is represented with the @ operator.

# 30. What does the backward method do?
backward method calculates the gradient of the function.

# 31. Why do we have to zero the gradients?
backward method calculates the gradient and store it internal variable grad. On subsequent call to backward, it calculates the new gradient and adds to the existing gradient. so we need to zero it before calculate gradient.

# 32. What information do we have to pass to Learner?
To create learner, it needs following parameters. 1. data loaders 2. model 3. optimization function 4. loss function 5. metrics

# 33. Show Python or pseudocode for the basic steps of a training loop?

# 34. What is ReLU? Draw a plot of it for values from -2 to +2.
```
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, steps=100, figsize=(6,4)):
    x = torch.linspace(min, max, steps)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, f(x))
    if title:
        ax.set_title(title)
    if tx:
        ax.set_xlabel(tx)
    if ty:
        ax.set_ylabel(ty)
    plt.show()

plot_function(F.relu)
```
![relu](/images/relu.png)


