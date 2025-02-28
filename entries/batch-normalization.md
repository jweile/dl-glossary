## Batch normalization

Batch Normalization is a technique that normalizes neuronal pre-activations per mini-batch. This avoids extremely large or small pre-actiations that would over- or under-saturate the activation function and thus lead to near-zero gradients. Batch normalization thus accelerates convergence by reducing internal covariate shift inside each batch. 

Batch normalization introduces additional parameters: a scale and an offset (which renders the bias term of the neuronal layer redunant).

Batch Normalization has been found to be very effective for Convolutional and Feedforward Neural Networks but hasn’t been successfully applied to Recurrent Neural Networks.

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
* [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)
* [Machine Learning Glossary: Batch Normalization](https://ml-cheatsheet.readthedocs.io/en/latest/layers.html#batchnorm)

