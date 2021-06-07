//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!

// Shay Green & Vinodh Kotipalli 2021

// Workaround for Clippy false positive in Rust 1.51.0.
// https://github.com/rust-lang/rust-clippy/issues/6546
#![allow(clippy::result_unit_err)]

use ndarray::prelude::*;

//use csv::*;
//use image::*;
//use plotters::prelude::*;

use std::collections::HashMap;
use thiserror::Error;

/// Errors during Model interaction.
#[derive(Error, Debug)]
pub enum ModelError<'a> {
    /// Model should be compiled before being used for training(fit) or prediction(predict)
    /// to make path separators easier.
    #[error("{0}: Model is not compiled")]
    ModelNotCompile(&'a str),

    #[error("{0}: Array dimension mismatch")]
    DimensionMismatch(&'a str),

    #[error("{0}: Input value outside expected range")]
    ValueNotInRange(&'a str),

    #[error("{0}: Activation function not implemented or not allowed for a given layer")]
    InvalidActivationFunction(&'a str),

    #[error("{0}: Loss function not implemented or not allowed for a given layer")]
    InvalidLossFunction(&'a str),
}

/// Result type for Model Errors errors.
pub type Result<'a, T> = std::result::Result<T, ModelError<'a>>;

/// Different types of activation functions supported by a NN Layer
#[derive(Clone)]
pub enum Activation<T: num_traits::float::Float> {
    /// Sigmoid Activation function
    Sigmoid,

    /// Tanh Activation function
    Tanh,

    /// Relu Activation function
    ReLu {
        alpha: T,
        max_value: T,
        threshold: T,
    },

    /// SoftMax Activation function
    SoftMax { axis: usize },
}

trait ActivationInterface<'a, T: num_traits::float::Float> {
    fn calculate_value(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;
    fn calculate_derivative(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;
}

/// helper functions
fn sigmoid<T: num_traits::float::Float>(x: T) -> T {
    T::from(1.0).unwrap() / (T::from(1.0).unwrap() + x.exp())
}

impl<'a, T: num_traits::float::Float> ActivationInterface<'a, T> for Activation<T> {
    fn calculate_value(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>> {
        match self {
            Activation::Sigmoid => Ok(inputs.mapv(sigmoid)),
            Activation::Tanh => Ok(inputs.mapv(|x| x.tanh())),
            Activation::ReLu {
                alpha,
                max_value,
                threshold,
            } => Ok(inputs.map(|x| {
                if *x > *max_value {
                    *max_value
                } else if *x > *threshold {
                    *x
                } else {
                    *alpha * (*x - *threshold)
                }
            })),
            Activation::SoftMax { axis } => Ok(inputs.mapv(|x| x.exp())
                / inputs
                    .mapv(|x| x.exp())
                    .sum_axis(Axis(*axis))
                    .insert_axis(Axis(*axis))),
        }
    }

    /// Partially Implemented -> ReLu and Softmax is pending. 
    fn calculate_derivative(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>> {
        match self {
            Activation::Sigmoid => Ok(inputs.mapv(|x| {
                let y = sigmoid(x);
                y * (T::from(1.0).unwrap() - y)
            })),
            Activation::Tanh => Ok(inputs.mapv(|x| (T::from(1.0).unwrap() - x.tanh().powf(T::from(2.0).unwrap())))),
            Activation::ReLu {
                alpha,
                max_value,
                threshold,
            } => Ok(inputs.map(|x| {
                if *x > *max_value {
                    *max_value
                } else if *x > *threshold {
                    *x
                } else {
                    *alpha * (*x - *threshold)
                }
            })),
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

trait LossFunction<'a, T: num_traits::float::Float> {
    fn calculate(&'a mut self, y_true: &'a Array2<T>, y_pred: &'a Array2<T>) -> Result<Array2<T>>;

    fn mean(&'a mut self, y_true: &'a Array2<T>, y_pred: &'a Array2<T>) -> Result<T>;

    fn mean_axis(&'a mut self, y_true: &'a Array2<T>, y_pred: &'a Array2<T>) -> Result<Array1<T>>;
}

/// Different types of Layers to construct a Neural Network
#[derive(Debug, Clone)]
pub enum Layer<'a, T: num_traits::float::Float> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: &'a str,
        input_dim: &'a u32,
        output_dim: &'a u32,
        weights: &'a Array2<T>,
        loss: &'a Array1<T>,
        prev_layer: Box<Option<Layer<'a, T>>>,
        next_layer: Box<Option<Layer<'a, T>>>,
    },
}

trait ConfigureLayer<'a, T: num_traits::float::Float> {
    fn default(&'a mut self) -> Result<()>;

    fn get_weights(&'a self) -> Result<Array2<T>>;

    fn set_weights(&'a mut self, weights: &'a Array2<T>) -> Result<()>;

    fn forward_propagate(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;

    fn back_propagate(&'a self, inputs: &'a Array2<T>) -> Result<()>;
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
pub enum ModelConstructor<'a, T: num_traits::float::Float> {
    /// Builds linear stack of layers into a model sequentially
    Sequential {
        name: &'a str,
        layers: Vec<Layer<'a, T>>,
    },
}

trait BuildModel<'a, T: num_traits::float::Float> {
    fn default(&'a mut self) -> Result<()>;
    fn add(&'a mut self, layer: &Layer<'a, T>) -> Result<()>;
    fn pop(&'a mut self, layer: &Layer<'a, T>) -> Result<()>;
    fn compile(
        &'a self,
        optimizer: &Optimizer<'a>,
        metrics: &[&'a str],
        validation_split: &'a f32,
    ) -> Result<Model<T>>;
}

/// Groups a linear stack of layers into a Model
#[derive(Debug, Clone)]
pub struct Model<'a, T: num_traits::float::Float> {
    pub name: &'a str,
    pub constructor: ModelConstructor<'a, T>,
    pub optimizer: Optimizer<'a>,
    pub metrics: Vec<&'a str>,
    pub validation_split: &'a T,
    pub history: HashMap<u32, HashMap<String, T>>,
}

trait UseModel<'a, T: num_traits::float::Float> {
    fn fit(&'a mut self, inputs: &'a Array2<T>, target: Array1<T>) -> Result<()>;

    fn predict(&'a self, inputs: &'a Array2<T>) -> Result<Array1<T>>;

    fn mse(&'a self, inputs: &'a Array2<T>, target: Array1<T>) -> Result<T>;

    fn entropy(&'a self, inputs: &'a Array2<T>, target: Array1<T>) -> Result<T>;

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

    #[test]
    fn test_tanh_activation() {
        fn f(x: f32) -> f32 {
            x.tanh()
        }

        let activation = Activation::Tanh;
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

    #[test]
    fn test_relu_activation() {
        fn f(x: f32, alpha: f32, max_value: f32, threshold: f32) -> f32 {
            let mut y = if x > threshold { x } else { alpha * x };
            if y > max_value {
                y = max_value
            };
            y
        }

        let mut alpha = 0.0;
        let mut max_value = f32::MAX;
        let mut threshold = 0.0;
        let mut activation = Activation::ReLu {
            alpha,
            max_value,
            threshold,
        };
        assert_eq!(
            activation.calculate(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0, 1.0]]).unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0], [1.0]]).unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![
                [
                    f(0.0, alpha, max_value, threshold),
                    f(1.0, alpha, max_value, threshold)
                ],
                [
                    f(2.0, alpha, max_value, threshold),
                    f(3.0, alpha, max_value, threshold)
                ]
            ]
        );
        alpha = 0.001;
        max_value = 1.0;
        threshold = 0.0;
        activation = Activation::ReLu {
            alpha,
            max_value,
            threshold,
        };
        assert_eq!(
            activation.calculate(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0, 1.0]]).unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation.calculate(&array![[0.0], [1.0]]).unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![
                [
                    f(0.0, alpha, max_value, threshold),
                    f(1.0, alpha, max_value, threshold)
                ],
                [
                    f(2.0, alpha, max_value, threshold),
                    f(3.0, alpha, max_value, threshold)
                ]
            ]
        );
    }
}
