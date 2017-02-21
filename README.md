## 500px Machine Learning Engineer Intern - Tech Challenge

This is my solution to "500px Machine Learning Engineer Intern - Tech Challenge".

#### Environment

* Python 3.6
* Tensorflow 1.0
* Numpy 1.11.3
* Matplotlib 2.0.0
* Seaborn 0.71

#### Description

Create adversarial images to fool a MNIST classifier in TensorFlow.

You can find experiments result in the `notebook` folder

In `src` folder, `generate_model.py` is used for training CNN model and dump model to target path

Usage:

```
python generate_model.py [--max_steps] [--learning_rate] [--dropout] [--data_dir]
                         [--log_dir] [--model_path]
```

`generate_adversarial.py` contains function that consumes a image list(img_list) and a desired class(target_class) , therefore produces the adversarial images that can fool the model.

<img src="/img/adversarial_versus.png" width="500">

some iteration processes:

![](/img/adversarial0_iter.png)

![](/img/adversarial4_iter.png)

#### To Do

* [x] Read [Karpathy's blog](http://karpathy.github.io/2015/03/30/breaking-convnets/)
* [x] Read paper [Deep Neural Networks are Easily Fooled](https://arxiv.org/abs/1412.1897) and [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [x] Go through all details in [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros)
* [x] Here is a [torch implement](https://github.com/e-lab/torch-toolbox/tree/master/Adversarial) for OVERFEAR network (Not so useful)
* [ ] Time to code ~
      * [x] Train a common CNN for MNIST
      * [x] generate adversarial images to misclassify any examples of ‘2’ as ‘6’
      * [x] generate adversarial images for any number as any number
      * [ ] *test SVM, RandomForest, GradientBoost Tree, deeper network for these adversarial images
      * [ ] *test my handwritten image
      * [ ] *analyze feature map for conv layer