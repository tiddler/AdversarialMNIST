## 500px Machine Learning Engineer Intern - Tech Challenge

---

This is my solution to "500px Machine Learning Engineer Intern - Tech Challenge".

#### Environment

* Python 3.6
* Tensorflow 1.0
* Numpy 1.11.3

#### Description

Create adversarial images to fool a MNIST classifier in TensorFlow.

#### To Do

* [x] Read [Karpathy's blog](http://karpathy.github.io/2015/03/30/breaking-convnets/)
* [x] Read paper [Deep Neural Networks are Easily Fooled](https://arxiv.org/abs/1412.1897) and [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [x] Go through all details in [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros)
* [x] Here is a [torch implement](https://github.com/e-lab/torch-toolbox/tree/master/Adversarial) for OVERFEAR network (Not so useful)
* [ ] Time to code ~
      * [x] Train a common CNN for MNIST
      * [ ] generate adversarial images to misclassify any examples of ‘2’ as ‘6’
      * [ ] generate adversarial images for any number as any number
      * [ ] 1 adversarial image vs 1 original image; 1 adversarial image vs 1 class images; 1 adversarial image vs all images;
      * [ ] *test SVM, RandomForest, GradientBoost Tree, deeper network for these adversarial images
      * [ ] *test my handwritten image
      * [ ] *analyze feature map for conv layer