## Capsule Network
A Capsule Neural Network (CapsNet) is a machine learning system that is a type of artificial neural network (ANN) that can be used to better model hierarchical relationships. The approach is an attempt to more closely mimic biological neural organization.

The idea is to add structures called "capsules" to a [convolutional neural network (CNN)](#cnn), and to reuse output from several of those capsules to form more stable (with respect to various perturbations) representations for higher capsules. The output is a vector consisting of the probability of an observation, and a pose for that observation. This vector is similar to what is done for example when doing classification with localization in CNNs.

* [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

