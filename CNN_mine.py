'''
TensorFlow 1.3
Python 3.6
By LiWenDi
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用ww = sess.run(h_conv2_show)表示绘制第二层卷积的结果，
#使用ww = sess.run(h_conv1_show)表示绘制第一层卷积的结果，
#绘制第二层请改train_first参数为False

train_first = False
STEPS = 1001 #训练次数


mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(STEPS):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("第 %d次训练，准确率为%g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})

print("-----------------------------------------------------------------------------------")
print("测试集上的准确率为： %g" % accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))

h_conv1_show = tf.nn.relu(conv2d(tf.reshape(mnist.test.images[0], [-1, 28, 28, 1]), W_conv1) + b_conv1) #改mnist.test.images[0]里面的[0]为别的数字来绘制其他图片
h_pool1_show = max_pool_2x2(h_conv1_show)
h_conv2_show = tf.nn.relu(conv2d(h_pool1_show, W_conv2) + b_conv2)


if train_first:
    channels = 32
    pic_size = 28
    ww = sess.run(h_conv1_show)
else:
    channels = 64
    pic_size = 14
    ww = sess.run(h_conv2_show)


ww = np.reshape(ww, [pic_size,pic_size,channels])
wws = []
for i in range(channels):
    wws.append(ww[:,:,i])
    wws[i] = np.reshape(wws[i], [pic_size,pic_size])

#画多个图的
fig, t = plt.subplots(channels//8,8)
for i in range(channels//8):
    for j in range(8):
        t[i][j].clear()
        t[i][j].set_axis_off()
        t[i][j].imshow(wws[i])
plt.draw()
plt.show()

#画单张的
plt.imshow(wws[i])
plt.show()