//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!

// Shay Green & Vinodh Kotipalli 2021

// Workaround for Clippy false positive in Rust 1.51.0.
// https://github.com/rust-lang/rust-clippy/issues/6546
#![allow(clippy::result_unit_err)]
#![allow(non_snake_case)]
use csv::*;
use image::*;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
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
#[derive(Clone)]
pub enum Activation<'a, S: ndarray::Data, D: ndarray::Dimension> {
    /// Sigmoid Activation function
    Sigmoid { x: &'a ArrayBase<S, D> },

    /// Tanh Activation function
    Tanh { x: &'a ArrayBase<S, D> },

    /// Relu Activation function
    ReLu {
        x: &'a ArrayBase<S, D>,
        alpha: &'a f32,
        max_value: &'a f32,
        threshold: &'a f32,
    },

    /// SoftMax Activation function
    SoftMax {
        x: &'a ArrayBase<S, D>,
        axis: &'a u32,
    },
}

/// Different types of loss functions supported by a NN Model
#[derive(Clone)]
pub enum Loss<'a, S: ndarray::Data, D: ndarray::Dimension> {
    /// Mean Square Error loss function
    MeanSquareError {
        y_true: &'a ArrayBase<S, D>,
        y_pred: &'a ArrayBase<S, D>,
    },

    /// Entropy loss function
    Entropy {
        y_true: &'a ArrayBase<S, D>,
        y_pred: &'a ArrayBase<S, D>,
    },
}

/// Different types of Layers to construct a Neural Network
#[derive(Debug, Clone)]
pub enum Layer<'a> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: &'a str,
        input_dim: &'a u32,
        output_dim: &'a u32,
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
