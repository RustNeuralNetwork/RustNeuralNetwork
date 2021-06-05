//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!

// Shay Green & Vinodh Kotipalli 2021

// Workaround for Clippy false positive in Rust 1.51.0.
// https://github.com/rust-lang/rust-clippy/issues/6546
#![allow(clippy::result_unit_err)]

use csv::*;
use image::*;
use ndarray::*;
use thiserror::Error;

/// Errors during Model interaction.
#[derive(Error, Debug)]
pub enum ModelError<'a> {
    /// Model should be compiled before being used for training(fit) or prediction(predict)
    /// to make path separators easier.
    #[error("{0}: Model is not compiled")]
    CompileError(&'a str),
}

/// Result type for Model Errors errors.
pub type Result<'a, T> = std::result::Result<T, ModelError<'a>>;

/// Different types of activation funtions supported by a NN Layer
#[derive(Debug, Clone)]
pub enum Activation<'a> {
    /// Sigmoid Activation function
    Sigmoid {},

    /// Tanh Activation function
    Tanh {},

    /// Relu Activation function
    ReLu {},

    /// SoftMax Activation function
    SoftMax {},
}

/// Different types of loss functions supported by a NN Model
#[derive(Debug, Clone)]
pub enum Loss<'a> {
    /// Mean Square Error loss function
    MeanSquareError {},

    /// Entropy loss function
    Entropy {},
}

/// Different types of Layers to construct a Neural Network
#[derive(Debug, Clone)]
pub enum Layer<'a> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: &'a str,
        input_dim: &'a i32,
        output_dim: &'a i32,
    },
}

/// Groups a linear stack of layers into a Model
#[derive(Debug, Clone)]
pub struct Sequential<'a> {
    pub name: &'a str,
    pub layers: Vec<Layer<'a>>,
}

/// Groups a linear stack of layers into a Model
#[derive(Debug, Clone)]
pub struct Model<'a> {
    pub name: &'a str,
    pub inputs: Layer<'a>,
    pub outputs: Layer<'a>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
