# RustNeuralNetwork

RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network for applying machine learning to simple classification problems.

by [Shay Green](shagreen@pdx.edu) and [Vinodh Kotipalli](vkotipa2@pdx.edu)

# Overview of Project Aspirations

RustNeuralNetwork allows you to design and specify a machine learning model using a fully connected, feed-forward neural network. This library allows you to instantiate a model and add multiple fully connected layers to construct your network. With each layer, you can specify the input and output dimensions, activation functions, loss functions, opitmizers, weights, and the previous and next layers. You can instantiate the model by creating a new model and adding or popping off individual layers. This allows you to specify the internal structure of your multi layer neural network. Once you have created your model, you can compile, train, and predict on the training and test data.

RustNeuralNetwork operates under the assumption that the training data given to the model is a NxM 2D array of input values with N input dimensions and M input examples that have already been preprocessed in the desired way along with a TxM 2D array of target values with T target output dimensions for each of the M examples.

The RustNeuralNetwork library allows you to forward propagate a single example through your model and back propagate the loss to update the model weights by calling the fit() function and then test the model after training by calling the predict() function on your test data.

# Our Goals and Progress

### We had two goals when building this library:

1. To complete training on the MNIST dataset:
   - In our implementation we were able to create and define the data and library interfaces for the RustNeuralNetwork library, but the full implementation is not yet complete. Currently, some Missing Implementation errors are still thrown when accessing library elements whose implementation was not completed.
2. To create a modular library:
   - We were able to create modular code by ensuring that the usage of objects like loss functions and optimizers were calls to their function implementations, that way we could ensure that the library could be expanded with additional forms of layers (convolutional, etc.), additional loss functions (binary cross entropy, etc.), as well as many other parameters that can typically be manipulated in a neural network.

### Our Approach:

1. Initially we came up with the library interface together based on the standard implementation of a fully connected neural network in Python. This allowed us to focus on code modularity and ensuring that our code could be expanded to encapsulate further complexity.
2. Vinodh focused mainly on the implementation of the library code starting with the loss and optimization functions and expanding to the functionality of individual layers. Later Vinodh was able to begin the implementation of the Model class to tie it all together, but the full implementation is not yet complete.
3. Shay focused mainly on the implementation example of the library using the MNIST dataset. This involved parsing the dataset into CSV format (via Python) to be read into Rust arrays for use in the binary executable example.

# How to Run

To compile library code, run:

```cmd
cargo build --lib
```

To run unit tests and doc tests in library, run:

```cmd
cargo test --lib
```

```cmd
cargo test --doc
```

To compile and run the binary executable MNIST classification example, you must first download the MNIST dataset [here](http://yann.lecun.com/exdb/mnist/) and place the four extracted ubyte files into the MNIST directory inside a new file labelled MNIST. The structure of your directory should look like:

```bash
├── MNIST
│   ├── MNIST
|   │   ├── t10k-images.idx3-ubyte
|   │   ├── t10k-labels.idx1-ubyte
|   │   ├── train-images.idx3-ubyte
|   │   └── train-labels.idx1-ubyte
│   └── main.py
├── src
│   ...
└── README.md
```

Then, inside the outermost MNIST directory, you can generate the csv format MNIST data by running the python script using:

```cmd
python3 main.py
```

Finally, run the binary crate using:

```cmd
cargo run --bin bin MNIST/mnist_test.csv MNIST/mnist_train.csv
```

# Testing and Implementation

This library was implemented using a basic neural network interface. There is an Activation enum that contains the possible activation functions. Currently the sigmoid, tanh, ReLU, and softmax activation functions are represented in the library. This structure allows us to implement additional activation functions that can be utilized in the models. Similarly, the library contains a Loss enum that contains the mean squared error and binary cross entropy loss functions. There is currently no BCE implementation, but like the activation functions, this enum structure allows us to add and implement additional loss functions. The RustNeuralNetwork also contains an enum for the Optimizer. We have implementation for the stochastic gradient descent optimizer, and the ability to continue to expand the optimizer options.

These three configurable parameters are then used in our next object. The Layer enum currently contains an implementation for fully connected layers, but has the capacity to be expanded to include convolutional layers, recurrent layers, etc. Our Layer implementation allows us to set the shape of the layer, get and set the layer weights, and forward propagate an example as well as backward propagate a loss.

The last piece of this puzzle is the model itself. For this implementation, we have a ModelConstructor enum that specifies how the model layers are concatenated. Currently there is implementation for sequential model construction with the potential to expand those options. This ModelConstructor has the ability to add and pop layers to configure your particular model architecture. Then, compiling the ModelConstructor returns a Model struct object that contains the particular architecture defined in the ModelConstructor. The Model struct contains methods to fit the model to a particular dataset and make a prediction based on input. Currently the fit and predict methods are not fully implemented.

The testing for this library is contained in the unit tests and doc tests that are in the [library file](https://github.com/RustNeuralNetwork/RustNeuralNetwork/blob/main/src/lib.rs) itself. A practical application of the RustNeuralNetwork can be viewed in the [binary file](https://github.com/RustNeuralNetwork/RustNeuralNetwork/blob/main/src/bin/bin.rs). Currently, since the library implementation is not fully complete, the binary execuatble file will throw exceptions, but it is meant to be an example of the potential application of the library in practice. In practice, the borrowing of values in our library is incorrect and therefore, the implementation of the library in the bin file does not work as intended.
