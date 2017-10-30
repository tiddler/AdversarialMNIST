## Note

#### [Karpathy's  blog](http://karpathy.github.io/2015/03/30/breaking-convnets/)

* wiggle is the gradient of layer
* perform a image update  instead of parameter update to reduce score for original label
* hold the model parameters fixed, compute the gradient of all pixels in the input image on target class
* High regularization gives smoother templates, but could be noticeably different (The reason is high regularization gives smaller weights, which means we need to change more dramatically in the original image)
* Lower regularization gives noisy templates but similar image

#### DeepFool: a simple and accurate method to fool deep neural networks

#### Adversarial Examples In The Physical World

#### Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples

* I was thinking that we can add noise in the gradient so that attacker can only get useless gradient. However, this paper says such "Gradient masking" is not so helpful, the attacker has a very good strategy for guessing where the holes in the defense are. The attacker can train their own model, a smooth model that has a gradient, make adversarial examples for their model, and then deploy those adversarial examples against our non-smooth model. 

#### Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images

#### Explaining and Harnessing Adversarial Examples

