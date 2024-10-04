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

