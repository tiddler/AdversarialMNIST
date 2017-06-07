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

There are two ways to fool a CNN model. The first one is raised by Ian Goodfellow (also known
as the fast gradient sign method), the other one is adopted by Papernot et al (also know as Jacobian-based saliency map). Here we first implement the easier one: fast gradient method.

This method can be simplified as

<img src="/img/fg_euqtion.png" width="250">

To conclude, the process is pretty straight-forward, we calculate the gradient w.r.t. target class and update input image based on the gradient. Repeat this process until we achieve a wanted confidence.

![](/img/process_1.png)

We tested two update strategies, one is to simply take the sign of gradient and perform uniformly updates for all pixels and the other is to use gradient value as parts of step size. It is easy to know that if the gradient of a pixel is large, we only need to add a little wiggle, which will bring enough influence on the prediction. Therefore, it will give us a less noisy adversarial image. We add a L2-norm -like loss to evaluate the Delta, a small loss means we add a very little perturbation to original image, making the adversarial image unnoticeable for humans. 

The experiment results support that -- updating by gradient value gives us a less L2-norm-like loss.

Here is the result for using the sign of gradient, the final L2-norm-like loss is 3.12143

![](/img/fg_sign_versus.png)

![](/img/fg_sign_iter.png)

Here is the result for using the value of gradient, the final L2-norm-like loss is 2.18394

![](/img/fg_grad_versus.png)

![](/img/fg_grad_iter.png)

As we want an adversarial image with unnoticeable perturbation, we perform gradient value updating.

You can find experiments result in the `notebook` folder

In `AdversarialMNIST` folder, `generate_model_old.py` is used for training CNN model and dump model to target path

Update:

As all functions have been packed into a class `AdversarialMNIST` as a package, you can use them easier

```python
import sys
# suppose current path is 'notebook/'
# set the package_path to the folder 'AdversarialMNIST'
package_path = '../'
sys.path.append(package_path)
from AdversarialMNIST.AdversarialMNIST import AdversarialMNIST

# create a object, the graph will be initialized
model = AdversarialMNIST()
# show the graph structure
model.show_trainable_variables()
"""
  layer_conv1_5x5_32/Weights:0
  layer_conv1/Bias:0
  layer_conv2_5x5x64/Weights:0
  layer_conv2/Bias:0
  layer_fc1_1024/Weights:0
  layer_fc1/Bias:0
  layer_fc2_10/Weights:0
  layer_fc2/Bias:0
"""
# Option 1: train a new model by MNIST and dump the model to target place *use default dataset
model.fit(learning_rate=1e-4, max_steps=20000, dropout=0.5, model_path='../model/MNIST.ckpt')

# Option 2: load previous model
model.load_model(model_path='../model/MNIST.ckpt')

# Option 3: train a new model by given dataset *use customized dataset
model.fit(x=new_train_images, y=new_train_labels, learning_rate=1e-4, 
          max_steps=5000, dropout=0.5, model_path='../model/MNIST_NEW.ckpt')

# make a prediction for images
# return predictions and probabilities for input images. Here shows the results for ordinary prediction 
normal_pred,normal_prob = model.predict(images, prob=True)

# generate adversarial images
# you can assign any target_class that you want to fool network with desired confidence
# if verbose is 1, it will generate all figures for report
ad_images = model.generate_adversarial(img_list=img_list, 
                                       target_class=7, eta=0.1, threshold=0.99, 
                              		save_path='../img/', file_name='adversarial', verbose=0)
#Predictions and probabilites of adv images                                  
adv_pred,adv_prob = model.predict(ad_images, prob=True)
```

Previous Usage:

```
python generate_model_old.py [--max_steps] [--learning_rate] [--dropout] [--data_dir]
                             [--log_dir] [--model_path]
```

`generate_adversarial_old.py` contains function that consumes a image list(img_list) and a desired class(target_class) , therefore produces the adversarial images that can fool the model.

This following image is the result generated by `generate_adversarial_old.py`

<img src="/img/adversarial_versus.png" width="500">

You can find iteration processes in the `img` folder

#### To Do

- [x] Read [Karpathy's blog](http://karpathy.github.io/2015/03/30/breaking-convnets/)
- [x] Read paper [Deep Neural Networks are Easily Fooled](https://arxiv.org/abs/1412.1897) and [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [x] Go through all details in [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros)
- [x] Here is a [torch implement](https://github.com/e-lab/torch-toolbox/tree/master/Adversarial) for OVERFEAT network (Not so useful)
- [x] Time to code ~
      * [x] Train a common CNN for MNIST
      * [x] generate adversarial images to misclassify any examples of ‘2’ as ‘6’
      * [x] generate adversarial images for any number as any number

- [ ] There seems to be much of fun to explore~ !

      - [x] Can these images fool other classifier? such as SVM, RandomForest, GradientBoost Tree, or a deeper network

            Actually, no.

      - [x] what if we add adversarial images in the training set? 

            Not so helpful as for defence result

      - [ ] How about one adversarial image for all images in the same class? i.e. a delta image that can apply to all images so that the CNN will still be fooled.

      - [ ] implement Jacobian-based saliency map described in [This paper](https://arxiv.org/abs/1511.07528)

      - [ ] All methods mentioned above need the architecture of network, however, recently, there is a black-box attack method, described in [this paper](https://arxiv.org/abs/1602.02697) 
