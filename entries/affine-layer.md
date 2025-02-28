## Affine layer

A fully-connected [layer](#layer) in a Neural Network. Affine means that each neuron in the previous layer is connected to each neuron in the current layer. In many ways, this is the “standard” layer of a Neural Network. Affine layers are often added on top of the outputs of [Convolutional Neural Networks](#cnn) or [Recurrent Neural Networks](#rnn) before making a final prediction. An affine layer is typically of the form `y = f(Wx + b)` where x are the layer inputs, W the parameters, b a bias vector, and f a nonlinear [activation function](#activation-function)

