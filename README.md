### RustNeuralNetwork

RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network for applying machine learning to simple classification problems.

by [Shay Green](shagreen@pdx.edu) and [Vinodh Kotipalli](vkotipa2@pdx.edu)

### Overview

RustNeuralNetwork allows you to design and specify a machine learning model using a fully connected, feed-forward neural network. This library allows you to instantiate a model and add multiple fully connected layers to construct your network. With each layer, you can specify the input and output dimensions, activation functions, loss functions, opitmizers, weights, and the previous and next layers. You can instantiate the model by creating a new model and adding or popping off individual layers. This allows you to specify the internal structure of your multi layer neural network. Once you have created your model, you can compile, train, and predict on the training and test data.

RustNeuralNetwork operates under the assumption that the training data given to the model is a NxM 2D array of input values with N input dimensions and M input examples that have already been preprocessed in the desired way along with a TxM 2D array of target values with T output dimensions for each of the M examples.

The RustNeuralNetwork library allows you to forward propagate a single example through your model and back propagate the loss to update the model weights by calling the fit() function and then test the model after training by calling the predict() function on your test data.

### Our Goals and Progress

# We had two goals when building this library:

1. To complete training on the MNIST dataset:
   In our implementation we were able to create and define the data and library interfaces for the RustNeuralNetwork library, but the full implementation is not yet complete. Currently, some Missing Implementation errors are still thrown when accessing library elements whose implementation was not completed.
2. To create a modular library:
   We were able to create modular code by ensuring that the usage of objects like loss functions and optimizers were calls to their function implementations, that way we could ensure that the library could be expanded with additional forms of layers (convolutional, etc.), additional loss functions (binary cross entropy, etc.), as well as many other parameters that can typically be manipulated in a neural network.

# Our Approach:

1. Initially we came up with the library interface together based on the standard implementation of a fully connected neural network in Python. This allowed us to focus on code modularity and ensuring that our code could be expanded to encapsulate further complexity.
2. Vinodh focused mainly on the implementation of the library code starting with the loss and optimization functions and expanding to the functionality of individual layers. Later Vinodh was able to begin the implementation of the Model class to tie it all together, but the full implementation is not yet complete.
3. Shay focused mainly on the implementation example of the library using the MNIST dataset. This involved parsing the dataset into CSV format (via Python) to be read into Rust arrays for use in the binary executable example.

### How to Run

To compile library code, run:
cargo build

To run unit tests and doc tests in library, run:
cargo test

To compile and run the binary executable MNIST classification example, run:
cargo run --bin bin MNIST/mnist_test.csv MNIST/mnist_train.csv

### Testing and Implementation
