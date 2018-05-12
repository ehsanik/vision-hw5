# CSE 455 Homework 5 #

Welcome friends,

It's time for neural networks! But this time with PyTorch! Yey! This homework might need a longer running time. Keep this in mind and start early.

PyTorch is a deep learning framework for fast, flexible experimentation. We are going to use it to train our classifiers.

For this homework you need to turn in `models.py`, `dataloder.py` and a PDF file `Report.pdf` including your plots and reasonings.

## 1. Installing PyTorch ##

You can either use [official PyTorch tutorial](https://pytorch.org/), [Our tutorial(Still getting updated)](https://github.com/ehsanik/pytorch_installation) 
or come to the workshop on Friday 11th, at 12pm in NAN181.
You can find a tutorial for MNIST [here](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py), a sample network for MNIST [here](https://github.com/pytorch/examples/blob/master/mnist/main.py), and the PyTorch slides are posted on the website.

## 2. Find the best network ##

In this part, we want to find a network architecture that can obtain reasonable results on Cifar-10 dataset. 
For the first steps you don't need to worry about reading data or calculating the accuracy. 
You are just going to implement the model structure and the forward pass.
Use `plot_progress.py` to compare methods and checkout how train and test accuracy changes. 
In the beginning and the end of each run of the program the address of the log
file for which the accuracies and losses are recorded is printed (It will look something like `logs/2018-05-09_14:39:36_log.txt`). You can use the following
command to plot the progress. 

```bash
python plot_progress.py --file_name <LOG FILE ADDRESS>
```

Note that you don't need to wait until the training is done to plot the progress.
Report the final plot for every model in your `Report.pdf` file. 
Compare how fast the models converge and how is the test accuracy changing during training.


## 2.1. Training a classifier using only one fully connected Layer ##

Implement a model to classify the images from Cifar-10 into ten categories using just one fully connected layer (Remember that fully connected layers are called Linear in PyTorch).
If you are new to PyTorch you may want to check out the tutorial on MNIST [here](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py).
Fill in the code for LazyNet in `models.py`. Run the model for 50 epoch and report the plots and accuracies: 

```bash
python main.py --model LazyNet --epochs 50
```

Analyze the behavior of your model (how well does it work?, how fast does it train?, why do you think it's working/not working?) 
and report the plots in your report file.

## 2.2. Training a classifier using multiple fully connected Layers ##

Implement a model for the same classification task using multiple fully connected layers. Start with a fully connected layer that maps the data from image size (32 * 32 * 3) to a vector of size 120, followed by another fully connected that reduces the size to 84 and finally a layer that maps the vector of size 84 to 10 classes.
Fill in the code for BoringNet in `models.py`. Run the model for 50 epoch and report the plots and accuracies 

```bash
python main.py --model BoringNet --epochs 50
```

Analyze the behavior of your model and report the plots in your report file.

### 2.2.1. Question ###

Try training this model with and without activations. How does the activations (such as ReLU) affect the training process and why?

## 2.3. Training a classifier using convolutions ##

Implement a model using convolutional, pooling and fully connected layers. Fill in the code for CoolNet in `models.py`. 
Run the model for 50 epoch and report the plots and accuracies 

```bash
python main.py --model CoolNet --epochs 50
```

Explain why you have chosen these layers and how they affected the performance. 
Analyze the behavior of your model and report the plots in your report file.

### 2.3.1. Question ###

Try using three different values for batch size. How do these values affect training and why?

For running the model using a custom batch size you can use:

```bash
python main.py --model CoolNet --epoch 50 --batchSize <Your Batch Size>
```

## 3. How does learning rate work? ##

When you are trying to train a neural network it is really hard to choose a proper learning rate. 
Try to train your model with different learning rates and plot the training accuracy, test accuracy and loss and compare the training progress 
for learning rates = 10, 0.1, 0.01, 0.0001.
Use command:

```bash
python main.py --lr <Learning Rate> --model CoolNet --epoch 50
```

Analyze the results and choose the best one. Why did you choose this value?

During training it is often useful to reduce learning rate as the training progresses (why?). 
Fill in `adjust_learning_rate` in `BaseModel` in `models.py` to reduce the learning rate by 10% every 50 epoch and observe the behavior of network for 150 epochs. 
Turn in your plots in `Report.pdf`.

```bash
python main.py --model CoolNet --epoch 150
```

## 4. Data Augmentation ##

Most of the popular computer vision datasets have tens of thousands of images. 
Cifar-10 is a dataset of 60000 32x32 colour images in 10 classes, which can be relatively small in compare to ImageNet which has 1M images. 
The more the number of parameters is, the more likely our model is to overfit to the small dataset. 
As you might have already faced this issue while training the CoolNet, after some iterations the training accuracy reaches its maximum (saturates)
while the test accuracy is still relatively low. 
To solve this problem, we use the data augmentation to help the network avoid overfitting.

Add data transformations in `dataloder.py` and compare the results. 
Just be aware that data augmentation should just happen during training phase. 
Run the following command with and without data augmentation and compare the result.

```bash
python3 main.py --epochs 200 --model CoolNet --lr 0.01
```

## 5. Change the loss function ##

Try Mean Squared Error loss instead of Cross Entropy and see how this affects the results and explain why you think this is happening. 

