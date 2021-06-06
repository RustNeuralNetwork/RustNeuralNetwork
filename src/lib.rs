//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!

// Shay Green & Vinodh Kotipalli 2021

// Workaround for Clippy false positive in Rust 1.51.0.
// https://github.com/rust-lang/rust-clippy/issues/6546
#![allow(clippy::result_unit_err)]

use ndarray::prelude::*;

use csv::*;
use image::*;
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

    #[error("{0}: Activation Function not implemented")]
    ActivationError(&'a str),
}

/// Result type for Model Errors errors.
pub type Result<'a, T> = std::result::Result<T, ModelError<'a>>;

/// Different types of activation functions supported by a NN Layer
#[derive(Clone)]
pub enum Activation {
    /// Sigmoid Activation function
    Sigmoid,

    /// Tanh Activation function
    Tanh,

    /// Relu Activation function
    ReLu {
        alpha: f32,
        max_value: f32,
        threshold: f32,
    },

    /// SoftMax Activation function
    SoftMax { axis: usize },
}

trait ActivationInterface<'a> {
    fn calculate(&'a self, inputs: &'a Array2<f32>) -> Result<Array2<f32>>;
}

impl<'a> ActivationInterface<'a> for Activation {
    fn calculate(&'a self, inputs: &'a Array2<f32>) -> Result<Array2<f32>> {
        match self {
            Activation::Sigmoid => Ok(inputs.mapv(|x| (1.0 / (1.0 + x.exp())))),
            Activation::Tanh => Ok(inputs.mapv(|x| x.tanh())),
            Activation::ReLu {
                alpha,
                max_value,
                threshold,
            } => Ok(inputs
                .map(|x| if *x > *threshold { *x } else { *alpha * *x })
                .map(|x| if *x < *max_value { *x } else { *max_value })),
            Activation::SoftMax { axis } => Ok(inputs.mapv(|x| x.exp())
                / inputs
                    .mapv(|x| x.exp())
                    .sum_axis(Axis(*axis))
                    .insert_axis(Axis(*axis))),
        }
    }
}

/// Different types of loss functions supported by a NN Model
#[derive(Clone)]
pub enum Loss {
    /// Mean Square Error loss function
    MeanSquareError,

    /// Entropy loss function
    Entropy,
}

trait LossFunction<'a> {
    fn calculate<L>(
        &'a mut self,
        y_true: &'a Array2<L>,
        y_pred: &'a Array2<L>,
    ) -> Result<Array2<f32>>;
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
        loss: &'a Array1<f32>,
    },
}

trait ConfigureLayer<'a> {
    fn default(&'a mut self) -> Result<()>;

    fn get_weights(&'a self) -> Result<Array2<f32>>;

    fn set_weights(&'a mut self, weights: &'a Array2<f32>) -> Result<()>;

    fn forward_propagate(&'a self, inputs: &'a Array2<f32>) -> Result<Array2<f32>>;

    fn back_propagate(&'a self, inputs: &'a Array2<f32>, errors: &'a Array2<f32>) -> Result<()>;
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
    fn fit<L>(&'a mut self, inputs: &'a Array2<f32>, target: Array1<L>) -> Result<()>;

    fn predict<L>(&'a self, inputs: &'a Array2<f32>, target: Array1<L>) -> Result<Array1<L>>;

    fn mse<L>(&'a self, inputs: &'a Array2<f32>, target: Array1<L>) -> Result<f32>;

    fn entropy<L>(&'a self, inputs: &'a Array2<f32>, target: Array1<L>) -> Result<f32>;

    fn mse_plot(&'a self) -> Result<()>;

    fn entropy_plot(&'a self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn test_sigmoid_activation() {
        fn f(x: f32) -> f32 {
            1.0 / (1.0 + x.exp())
        }
        let activation = Activation::Sigmoid;
        assert_eq!(
            activation.calculate(&array![[0.0]]).unwrap(),
            &array![[f(0.0)]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0, 1.0]]).unwrap(),
            &array![[f(0.0), f(1.0)]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0], [1.0]]).unwrap(),
            &array![[f(0.0)], [f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)], [f(2.0), f(3.0)]]
        );
    }
}
