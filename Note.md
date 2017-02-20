## Note

#### [Karpathy's  blog](http://karpathy.github.io/2015/03/30/breaking-convnets/)

* wiggle is the gradient of layer
* perform a image update  instead of parameter update to reduce score for original label
* hold the model parameters fixed, compute the gradient of all pixels in the input image on target class
* High regularization gives smoother templates, but could be noticeably different (The reason is high regularization gives smaller weights, which means we need to change more dramatically in the original image)
* Lower regularization gives noisy templates but similar image