//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!

// Shay Green & Vinodh Kotipalli 2021

// Workaround for Clippy false positive in Rust 1.51.0.
// https://github.com/rust-lang/rust-clippy/issues/6546
#![allow(clippy::result_unit_err)]

use csv::*;
use image::*;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use plotters::prelude::*;

use std::collections::HashMap;
use thiserror::Error;

/// Errors during Model interaction.
#[derive(Error, Debug)]
pub enum ModelError<'a> {
    /// Model should be compiled before being used for training(fit) or prediction(predict)
    /// to make path separators easier.
    #[error("{0}: Model is not compiled")]
    CompileError(&'a str),

    #[error("{0}: Array dimension mismatch")]
    DimensionError(&'a str),

    #[error("{0}: Input value outside expected range")]
    ValueRangeError(&'a str),
}

/// Result type for Model Errors errors.
pub type Result<'a, T> = std::result::Result<T, ModelError<'a>>;

/// Different types of activation functions supported by a NN Layer
#[derive(Clone)]
pub enum Activation<'a, S: ndarray::Data, D: ndarray::Dimension> {
    /// Sigmoid Activation function
    Sigmoid { result: &'a ArrayBase<S, D> },

    /// Tanh Activation function
    Tanh { result: &'a ArrayBase<S, D> },

    /// Relu Activation function
    ReLu {
        result: &'a ArrayBase<S, D>,
        alpha: &'a f32,
        max_value: &'a f32,
        threshold: &'a f32,
    },

    /// SoftMax Activation function
    SoftMax {
        result: &'a ArrayBase<S, D>,
        axis: &'a u32,
    },
}

trait ActivationFunction<'a, S: ndarray::Data, D: ndarray::Dimension> {
    fn default(&'a mut self) -> Result<()>;

    fn calculate<Si: ndarray::Data, Di: ndarray::Dimension>(
        &'a mut self,
        x: &'a ArrayBase<Si, Di>,
    ) -> Result<ArrayBase<S, D>>;
}

/// Different types of loss functions supported by a NN Model
#[derive(Clone)]
pub enum Loss<'a> {
    /// Mean Square Error loss function
    MeanSquareError { result: &'a f32 },

    /// Entropy loss function
    Entropy { result: &'a f32 },
}

trait LossFunction<'a, S: ndarray::Data, D: ndarray::Dimension> {
    fn default(&'a mut self) -> Result<()>;

    fn calculate<L>(
        &'a mut self,
        y_true: &'a Array1<L>,
        y_pred: &'a Array1<L>,
    ) -> Result<ArrayBase<S, D>>;
}

/// Different types of Layers to construct a Neural Network
#[derive(Debug, Clone)]
pub enum Layer<'a> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: &'a str,
        input_dim: &'a u32,
        output_dim: &'a u32,
        weights: &'a Array2<f32>,
    },
}

trait ConfigureLayer<'a> {
    fn default(&'a mut self) -> Result<()>;

    fn get_weights(&'a self) -> Result<Array2<f32>>;

    fn set_weights(&'a mut self, weights: &'a Array2<f32>) -> Result<()>;
}

/// Different types of Optimizers functions
#[derive(Debug, Clone)]
pub enum Optimizer<'a> {
    /// Builds linear stack of layers into a model sequentially
    StochasticGradientDescent {
        learning_rate: &'a f32,
        momentum: &'a f32,
    },
}

trait OptimizerFunction<'a> {
    fn default(&'a mut self) -> Result<()>;

    fn get_params<K, V>(&'a self) -> Result<HashMap<K, V>>;

    fn set_params<K, V>(&'a self, values: &'a HashMap<K, V>) -> Result<()>;
}

/// Different types of NN Model Constructors
#[derive(Debug, Clone)]
pub enum ModelConstructor<'a> {
    /// Builds linear stack of layers into a model sequentially
    Sequential {
        name: &'a str,
        layers: Vec<Layer<'a>>,
    },
}

trait BuildModel<'a> {
    fn default(&'a mut self) -> Result<()>;
    fn add(&'a mut self, layer: &Layer<'a>) -> Result<()>;
    fn pop(&'a mut self, layer: &Layer<'a>) -> Result<()>;
    fn compile(
        &'a self,
        optimizer: &Optimizer<'a>,
        metrics: &[&'a str],
        validation_split: &'a f32,
    ) -> Result<Model>;
}

/// Groups a linear stack of layers into a Model
#[derive(Debug, Clone)]
pub struct Model<'a> {
    pub name: &'a str,
    pub constructor: ModelConstructor<'a>,
    pub optimizer: Optimizer<'a>,
    pub metrics: Vec<&'a str>,
    pub validation_split: &'a f32,
    pub history: HashMap<u32, HashMap<String, f32>>,
}

trait UseModel<'a> {
    fn fit<S: ndarray::Data, D: ndarray::Dimension, L>(
        &'a mut self,
        inputs: &'a ArrayBase<S, D>,
        target: Array1<L>,
    ) -> Result<()>;

    fn predict<S: ndarray::Data, D: ndarray::Dimension, L>(
        &'a self,
        inputs: &'a ArrayBase<S, D>,
        target: Array1<L>,
    ) -> Result<Array1<L>>;

    fn mse<S: ndarray::Data, D: ndarray::Dimension, L>(
        &'a self,
        inputs: &'a ArrayBase<S, D>,
        target: Array1<L>,
    ) -> Result<f32>;

    fn entropy<S: ndarray::Data, D: ndarray::Dimension, L>(
        &'a self,
        inputs: &'a ArrayBase<S, D>,
        target: Array1<L>,
    ) -> Result<f32>;

    fn mse_plot(&'a self) -> Result<()>;

    fn entropy_plot(&'a self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
