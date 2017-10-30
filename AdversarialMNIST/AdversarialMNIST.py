# Copyright 2017 Ruifan Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This is the class for adversarial image generation

It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AdversarialMNIST:
  def __init__(self):
    self.x = tf.placeholder(tf.float32, shape=[None, 784])
    self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

    self.W_conv1 = self.weight_variable([5, 5, 1, 32], name='layer_conv1_5x5_32/Weights')
    self.b_conv1 = self.bias_variable([32], name='layer_conv1/Bias')

    self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
    self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)

    # Second conv layer with a pool layer
    self.W_conv2 = self.weight_variable([5, 5, 32, 64], name='layer_conv2_5x5x64/Weights')
    self.b_conv2 = self.bias_variable([64], name='layer_conv2/Bias')

    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
    self.h_pool2 = self.max_pool_2x2(self.h_conv2)

    # First Full-connect layer
    self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='layer_fc1_1024/Weights')
    self.b_fc1 = self.bias_variable([1024], name='layer_fc1/Bias')

    self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

    self.keep_prob = tf.placeholder(tf.float32)
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    # Second Full-connect layer
    self.W_fc2 = self.weight_variable([1024, 10], name='layer_fc2_10/Weights')
    self.b_fc2 = self.bias_variable([10], name='layer_fc2/Bias')

    # output layer
    self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
    self.y_pred = tf.nn.softmax(self.y_conv)
    self.cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

    self.sess = tf.InteractiveSession()
    self.mnist = None

  def show_trainable_variables(self):
    for var in tf.trainable_variables():
      print(var.name)

  def weight_variable(self, shape, name):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

  def bias_variable(self, shape, name):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

  def conv2d(self, x, W):
    """simple conv2d layer"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    """a simple 2x2 max pool layer"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  def load_mnist(self, mnist_path='/tmp/tensorflow/mnist/input_data'):
    """load mnist data from given path"""
    self.mnist = input_data.read_data_sets(mnist_path, one_hot=True)

  def fit(self, x=None, y=None, batch_size=50, learning_rate=1e-4,
          max_steps=20000, dropout=0.5, model_path=''):
    """provide pipeline-like api, also know as training

    Args:
      x: `numpy.array`, training images
      y: `numpy.array`, training labels
      batch_size: `int`, batch size for training, default: 50
      learning_rate: `float`, learning rate for training, default: 1e-4
      max_steps: `int`, max training rounds, default: 20000
      dropout: `float`, dropout rate for fc layer, default: 0.5
      model_path: `String`, if set, dump model to target path, default: ''

    Returns:
      None

    Example:
      >>> self.fit(x=images, y=labels, batch_size=50, learning_rate=1e-4,
                    max_steps=20000, dropout=0.5, model_path='../model/MNIST.ckpt')
    """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    self.sess.run(tf.global_variables_initializer())

    if x is not None and y is not None:
      i = 0
      for iter in range(0, max_steps):
        batch = (x[i: i+batch_size], y[i: i+batch_size])
        i = (i+batch_size) % len(x)

        if iter % 1000 == 0:
          train_accuracy = self.accuracy.eval(feed_dict={
              self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
          print('step %d, training accuracy %g' % (iter, train_accuracy))
        train_step.run(feed_dict={self.x: batch[0],
                        self.y_: batch[1], self.keep_prob: dropout})

    else:
      if self.mnist == None:
        self.load_mnist()
      for i in range(max_steps):
        batch = self.mnist.train.next_batch(batch_size)
        if i % 500 == 0:
          train_accuracy = self.accuracy.eval(feed_dict={
              self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
          print('step %d, training accuracy %g'%(i, train_accuracy))
        train_step.run(feed_dict={self.x: batch[0],
                        self.y_: batch[1], self.keep_prob: dropout})

      print('test accuracy %g' % self.accuracy.eval(feed_dict={
          self.x: self.mnist.test.images, self.y_: self.mnist.test.labels, self.keep_prob: 1.0}))

    if len(model_path):
      model_path = model_path
      save_path = tf.train.Saver().save(self.sess, model_path)
      print('Model saved in file: ', save_path)

  def load_model(self, model_path='../model/MNIST.ckpt'):
    """load model from given path"""
    tf.train.Saver().restore(self.sess, model_path)
    print('load model from', model_path)

  def predict(self, input, prob=True):
    """provide pipeline-like api, return prediction and probability"""
    prediction = tf.argmax(self.y_pred, 1)
    prediction_val = prediction.eval(feed_dict={self.x: input, self.keep_prob: 1.0}, session=self.sess)
    probabilities_val = None
    if prob:
      probabilities = self.y_pred
      probabilities_val = probabilities.eval(feed_dict={self.x: input, self.keep_prob: 1.0}, session=self.sess)
    return (prediction_val, probabilities_val)

  def generate_adversarial(self, img_list, target_class, eta=0.005,
        threshold=0.99, save_path=None, file_name='adversarial', verbose=0):
    """generate adversarial images, note that the model should be loaded
    or fitted(trained) previously

    Args:
      img_list: `string`, the img list that need to generate adversarial images
      target_class: `int`, the wanted label
      eta: `float`, learning rate (or step size), default: 0.005
      threshold: `float`, the confidence we want to fool, default: 0.99 (99%)
      save_path: `string`, the path to img/ folder
      file_name: `string`, the name for saving file, default:'adversarial'
      verbose: `int`, verbose=0, omit the training graphs, default: 0

    Returns:
      `np.array`: the final adversarial image for each img in img_list

    Example:
      >>> self.generate_adversarial(
                  img_list=img_list, target_class=6, eta=0.01, threshold=0.99,
                  save_path='../img/', file_name='adversarial', verbose=1)
      np.ndarray(...)
    """
    prediction = tf.argmax(self.y_pred, 1)
    probabilities = self.y_pred

    # use tf.gradients() to compute the gradient for each pixel
    img_gradient = tf.gradients(self.cross_entropy, self.x)[0]

    # the returning imgs
    adversarial_img_list = list()

    # generate versus figure
    if verbose == 1:
      sns.set_style('white')
      versus_fig = plt.figure(figsize=(9, 40))

    # in case user pass in a str 
    target_class = int(target_class)

    # iter over each img
    for img_index in range(0, img_list.shape[0]):
      if img_index % 100 == 0:
        print('generate for', img_index, 'images')
      adversarial_img = img_list[img_index: img_index+1].copy()
      adversarial_label = np.zeros((1, 10))
      adversarial_label[:, target_class] = 1
      
      confidence = 0
      iter_num = 0
      prob_history = list()
      while confidence < threshold:
        probabilities_val = probabilities.eval(feed_dict=
                          {self.x: adversarial_img, self.keep_prob: 1.0}, session=self.sess)
        confidence = probabilities_val[:, target_class]
        prob_history.append(probabilities_val[0])
        
        gradient = img_gradient.eval(
            {self.x: adversarial_img, self.y_: adversarial_label, self.keep_prob: 1.0})
        adversarial_img -= eta * gradient
        # limit the value to [0, 1]
        adversarial_img = np.clip(adversarial_img, 0, 1)
        iter_num += 1
      adversarial_img_list.append(adversarial_img)
      
      if verbose != 0:
        print('generate adversarial image after', iter_num, 'iterations')
        # generate versus figure

        ax1 = versus_fig.add_subplot(10, 3, 3*img_index+1)
        ax1.axis('off')
        ax1.imshow(img_list[img_index].reshape([28, 28]), 
                  interpolation=None, cmap=plt.cm.gray)
        ax1.title.set_text(
            'Confidence for origin: ' + '{:.4f}'.format(np.amax(prob_history[0])) 
            + '\nConfidence for ' + str(target_class)+ 
            ': ' + '{:.4f}'.format(prob_history[0][target_class]))

        ax2 = versus_fig.add_subplot(10, 3, 3*img_index+2)
        ax2.axis('off')
        ax2.imshow((adversarial_img - img_list[img_index]).reshape([28, 28]),
                    interpolation=None, cmap=plt.cm.gray)
        ax2.title.set_text('Delta')

        ax3 = versus_fig.add_subplot(10, 3, 3*img_index+3)
        ax3.axis('off')
        ax3.imshow((adversarial_img).reshape([28, 28]), 
                    interpolation=None, cmap=plt.cm.gray)
        ax3.title.set_text(
            'Confidence for origin: ' + '{:.4f}'.format(np.amax(prob_history[-1]))
            + '\nConfidence for ' + str(target_class)+ 
            ': ' + '{:.4f}'.format(prob_history[-1][target_class]))

        print("Difference Measure:", 
                        np.sum((adversarial_img - img_list[img_index]) ** 2))
        sns.set_style('whitegrid')
        colors_list = sns.color_palette("Paired", 10)
        # generate Iteration figure
        prob_history = np.array(prob_history)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for i, record in enumerate(prob_history.T):
            plt.plot(record, color=colors_list[i])

        ax.legend([str(x) for x in range(0, 10)], 
                    loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Prediction Confidence')
        fig.savefig(save_path + file_name + str(img_index) + '_iter.png')
    
    if verbose != 0:
      versus_fig.tight_layout()
      versus_fig.savefig(save_path + file_name + '_versus.png')
    return np.squeeze(np.array(adversarial_img_list))
  
  # Under construction 

  # def generate_general_adversarial(self, x=None, target_class=6, 
  #                       max_steps=1000, batch_size=50, dropout=0.5):
  #   self.x_ad = self.weight_variable([784], name='x_ad')
  #   x_ad_image = tf.add(self.x_image, self.x_ad)
  #   x_ad_image = tf.clip_by_value(x_ad_image, 0.0, 1.0)
    
  #   # wait to see whether it works
  #   temp = self.x_image
  #   self.x_image = x_ad_image

  #   l2_weight = 0.02
  #   l2_loss = l2_weight * tf.nn.l2_loss(self.x_ad)
  #   loss_adversary = self.cross_entropy + l2_loss
  #   self.optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-4)
  #   self.generate_step = self.optimizer_adversary.minimize(self.cross_entropy, var_list=[self.x_ad])

  #   init_list = ['x_ad:0', 'beta1_power:0', 'beta2_power:0', 'x_ad/Adam:0', 'x_ad/Adam_1:0']
  #   self.sess.run(tf.variables_initializer([var for var in tf.global_variables() if var.name in init_list]))


  #   i = 0
  #   for iter in range(0, max_steps):
  #     i = (i+batch_size) % len(x)
  #     adversarial_label = np.zeros((len(x[i: i+batch_size]), 10))
  #     adversarial_label[:, target_class] = 1
  #     self.sess.run(self.generate_step, feed_dict={self.x: x[i: i+batch_size],
  #                         self.y_: adversarial_label, self.keep_prob: dropout})
  #     ad_delta = self.sess.run(self.x_ad)
  #     if iter % 100 == 0:
  #       print('l2_loss', self.sess.run(l2_loss))
  #       print('cross_loss', self.sess.run(self.cross_entropy, feed_dict={self.x: x[i: i+batch_size],
  #                         self.y_: adversarial_label, self.keep_prob: dropout}))
  #       print('combine_loss', self.sess.run(loss_adversary, feed_dict={self.x: x[i: i+batch_size],
  #                         self.y_: adversarial_label, self.keep_prob: dropout}))
    
  #   # wait to see whether it works 
  #   self.x_image = temp
  #   return np.squeeze(ad_delta)