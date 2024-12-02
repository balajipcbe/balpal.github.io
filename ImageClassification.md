#  Questionnaire

## 1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
   Performance and image data accuracy are the primary factors for resizing to a larger size on CPU then to a smaller size on the GPU.
   For performance reason, all images need to be in uniform size to create tensors. To make it as uniform, the image crop might introduce spurious empty zone and degrade the data.
   and compose all data augmentations in a batch. Uniform image size, batch data augmentation increases GPU performance.

## 2. If you are not familiar with regular expressions, find a regular expression tutorial and some problem sets, and complete them. Have a look on the book's website for suggestions.

   
    
