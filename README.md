# CNN_basic

## Introduction

有同学找我绘制出CNN卷积途中的图像，遂做了这个东西，mnist数据集和两个基本的卷积。可以绘制一张图片经过一次卷积而成的32个通道或者两次卷积而成的64个通道的图片。单通道绘制。

 - It's a CNN neural network on mnist dataset, with showing the convolutional picture available. 

 - Its structure is two basic convolutional neural networks. This python code can draw either the first one or the second one by one channel. The first one has 32 channels and the second has 64.

## How to use

 - You just need to change the switch `` train_first `` to choose which convolutional neural network to draw. And change the 
`` tf.reshape(mnist.test.images[0], [-1, 28, 28, 1]) `` to change which number to draw.

![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/220.jpg?raw=true)

![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/221.jpg?raw=true)

 - The result is like this:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/219.jpg?raw=true)
 _The first convolutional neural network_
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/218.jpg?raw=true)
_The second convolutional neural network_
