## Max-Pooling
A [pooling](#pooling) operations typically used in Convolutional Neural Networks. A max-pooling layer selects the maximum value from a patch of features. Just like a convolutional layer, pooling layers are parameterized by a window (patch) size and stride size. For example, we may slide a window of size 2×2 over a 10×10 feature matrix using stride size 2, selecting the max across all 4 values within each window, resulting in a new 5×5 feature matrix.

Pooling layers help to reduce the dimensionality of a representation by keeping only the most salient information, and in the case of image inputs, they provide basic invariance to translation (the same maximum values will be selected even if the image is shifted by a few pixels). Pooling layers are typically inserted between successive convolutional layers.

