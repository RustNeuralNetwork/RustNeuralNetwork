//! RustNeuralNetwork is Neural Network library that supports multi-layer full connected neural network
//! for applying machine learning to simple classification problems.
//!
//! This version of the library build primarily focus on NN structure, built in a modular.
//! We used different Structs and Enums to have comparable interface as TensorFlow in python.
//! This is very basic version but we tried to keep our architecture modular to allow for future expansion.
//! For instance we defined Activation function as its own Enum type and implemented required methods using pattern matching,
//! so that we can expand the support more variety of function types.
//! Similar idea is used for Layer enum, which now only supports Dense layer required for NN, but can expand to layers required for other models like CNN.
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

    #[error("{0}: Activation function not allowed for a given layer")]
    InvalidActivationFunction(&'a str),

    #[error("{0}: Loss function  not allowed for a given layer")]
    InvalidLossFunction(&'a str),

    #[error("{0}: Feature is defined but not implemented")]
    MissingImplementation(&'a str),
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

/// Set of Helper functions to refactor the Trait implementations for better readability

/// Sigmoid function that takes in and returns a generic float type
fn sigmoid<T: num_traits::float::Float>(x: T) -> T {
    T::from(1.0).unwrap() / (T::from(1.0).unwrap() + x.exp())
}

fn relu<T: num_traits::float::Float>(x: T, alpha: T, max_value: T, threshold: T) -> T {
    if x > max_value {
        max_value
    } else if x > threshold {
        x
    } else {
        alpha * (x - threshold)
    }
}

fn relu_derivative<T: num_traits::float::Float>(x: T, alpha: T, max_value: T, threshold: T) -> T {
    if x > max_value {
        T::from(0.0).unwrap()
    } else if x == max_value {
        T::from(0.5).unwrap()
    } else if x > threshold {
        T::from(1.0).unwrap()
    } else if x == threshold {
        (T::from(1.0).unwrap() + alpha) / T::from(2.0).unwrap()
    } else {
        alpha
    }
}

fn max_axis<T: num_traits::float::Float>(axis: usize, inputs: &'_ Array2<T>) -> Array2<T> {
    inputs
        .map_axis(Axis(axis), |x| {
            let first = x.first();
            let z = x
                .fold(first, |acc, y| if acc > Some(y) { acc } else { Some(y) })
                .unwrap();
            z.to_owned()
        })
        .insert_axis(Axis(axis))
}

fn _min_axis<T: num_traits::float::Float>(axis: usize, inputs: &'_ Array2<T>) -> Array2<T> {
    inputs
        .map_axis(Axis(axis), |x| {
            let first = x.first();
            let z = x
                .fold(first, |acc, y| if acc > Some(y) { acc } else { Some(y) })
                .unwrap();
            z.to_owned()
        })
        .insert_axis(Axis(axis))
}

fn mean_axis<T: num_traits::float::Float>(axis: usize, inputs: &'_ Array2<T>) -> Array1<T> {
    inputs.map_axis(Axis(axis), |x| x.sum() / T::from(x.len()).unwrap())
}

fn stable_softmax<T: num_traits::float::Float>(axis: usize, inputs: &'_ Array2<T>) -> Array2<T> {
    let inputs_shifted = inputs - max_axis(axis, inputs);
    inputs_shifted.mapv(|x| x.exp())
        / inputs_shifted
            .mapv(|x| x.exp())
            .sum_axis(Axis(axis))
            .insert_axis(Axis(axis))
}

impl<'a, T: num_traits::float::Float> ActivationInterface<'a, T> for Activation<T> {
    /// Calculates activation values using multiple functions used in NN base ML models
    fn calculate_value(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>> {
        match self {
            Activation::Sigmoid => Ok(inputs.mapv(sigmoid)),
            Activation::Tanh => Ok(inputs.mapv(|x| x.tanh())),
            Activation::ReLu {
                alpha,
                max_value,
                threshold,
            } => Ok(inputs.mapv(|x| relu(x, *alpha, *max_value, *threshold))),
            Activation::SoftMax { axis } => Ok(stable_softmax(*axis, inputs)),
        }
    }

    /// Calculates derivatives of activation values using multiple functions used in NN base ML models
    fn calculate_derivative(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>> {
        match self {
            Activation::Sigmoid => Ok(inputs.mapv(|x| {
                let y = sigmoid(x);
                y * (T::from(1.0).unwrap() - y)
            })),
            Activation::Tanh => {
                Ok(inputs.mapv(|x| (T::from(1.0).unwrap() - x.tanh().powf(T::from(2.0).unwrap()))))
            }
            Activation::ReLu {
                alpha,
                max_value,
                threshold,
            } => Ok(inputs.mapv(|x| relu_derivative(x, *alpha, *max_value, *threshold))),
            Activation::SoftMax { axis } => {
                Ok(stable_softmax(*axis, inputs).mapv(|x| x * (T::from(1.0).unwrap() - x)))
            }
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

trait LossInterface<'a, T: num_traits::float::Float> {
    fn calculate_value(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
    ) -> Result<Array2<T>>;

    fn calculate_derivative(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
    ) -> Result<Array2<T>>;

    fn mean(&'a mut self, y_true: &'a Array2<T>, y_pred: &'a Array2<T>) -> Result<T>;

    fn mean_axis(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
        axis: usize,
    ) -> Result<Array1<T>>;
}

/// As of now library only supports MSE loss and not entropy.
impl<'a, T: num_traits::float::Float> LossInterface<'a, T> for Loss {
    /// Calculates loss values using multiple functions used in NN base ML models
    fn calculate_value(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
    ) -> Result<Array2<T>> {
        match self {
            Loss::MeanSquareError => Ok((y_pred - y_true).mapv(|x| (x * x))),
            Loss::Entropy => Err(ModelError::MissingImplementation("Entropy")),
        }
    }

    /// Calculates loss values using multiple functions used in NN base ML models
    fn calculate_derivative(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
    ) -> Result<Array2<T>> {
        match self {
            Loss::MeanSquareError => Ok(y_pred - y_true),
            Loss::Entropy => Err(ModelError::MissingImplementation("Entropy")),
        }
    }

    /// Calculates mean of loss values using multiple functions used in NN base ML models
    fn mean(&'a mut self, y_true: &'a Array2<T>, y_pred: &'a Array2<T>) -> Result<T> {
        match self {
            Loss::MeanSquareError => {
                let mse = (y_pred - y_true).mapv(|x| (x * x));
                Ok(mse.sum() / T::from(mse.len()).unwrap())
            }
            Loss::Entropy => Err(ModelError::MissingImplementation("Entropy")),
        }
    }

    /// Calculates mean of loss values over given axis using multiple functions used in NN base ML models
    fn mean_axis(
        &'a mut self,
        y_true: &'a Array2<T>,
        y_pred: &'a Array2<T>,
        axis: usize,
    ) -> Result<Array1<T>> {
        match self {
            Loss::MeanSquareError => Ok(mean_axis(axis, &(y_pred - y_true).mapv(|x| (x * x)))),
            Loss::Entropy => Err(ModelError::MissingImplementation("Entropy")),
        }
    }
}
/// Different types of Layers to construct a Neural Network
#[derive(Debug, Clone)]
pub enum Layer<'a, T: num_traits::float::Float> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: &'a str,
        input_dim: u32,
        output_dim: u32,
        weights: &'a Array2<T>,
        loss_values: &'a Option<Array2<T>>,
        prev_layer: Box<Option<Layer<'a, T>>>,
        next_layer: Box<Option<Layer<'a, T>>>,
    },
}

trait ConfigureLayer<'a, T: num_traits::float::Float> {
    fn create(&'a mut self) -> Result<()>;

    fn shape(&'a self) -> Vec<u32>;

    fn get_weights(&'a self) -> Result<Array2<T>>;

    fn set_weights(&'a mut self, weights: &'a Array2<T>) -> Result<()>;

    fn forward_propagate(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;

    fn back_propagate(&'a self, inputs: &'a Array2<T>) -> Result<()>;
}

impl<'a, T: num_traits::float::Float> ConfigureLayer<'a, T> for Layer<'a, T> {
    fn create(&'a mut self) -> Result<()> {
        match self {
            Layer::Dense {
                activation: _,
                input_dim,
                output_dim,
                weights: _,
                loss_values: _,
                prev_layer,
                next_layer,
            } => {
                let layer_below = prev_layer.to_owned();
                let layer_above = next_layer.to_owned();
                if (layer_below.is_none() || layer_below.unwrap().shape()[1] == *input_dim)
                    && (layer_above.is_none() || layer_above.unwrap().shape()[0] == *output_dim)
                {
                    Ok(())
                } else {
                    Err(ModelError::DimensionMismatch("Entropy"))
                }
            }
        }
    }

    fn shape(&'a self) -> Vec<u32> {
        match self {
            Layer::Dense {
                activation: _,
                input_dim,
                output_dim,
                weights: _,
                loss_values: _,
                prev_layer: _,
                next_layer: _,
            } => [input_dim.to_owned(), output_dim.to_owned()].to_vec(),
        }
    }

    fn get_weights(&'a self) -> Result<Array2<T>> {
        Err(ModelError::MissingImplementation(
            "ConfigureLayer::get_weights",
        ))
    }

    fn set_weights(&'a mut self, weights: &'a Array2<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "ConfigureLayer::set_weights",
        ))
    }

    fn forward_propagate(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>> {
        Err(ModelError::MissingImplementation(
            "ConfigureLayer::forward_propagate",
        ))
    }

    fn back_propagate(&'a self, inputs: &'a Array2<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "ConfigureLayer::back_propagate",
        ))
    }
}

/// Different types of Optimizers functions
#[derive(Debug, Clone)]
pub enum Optimizer<'a> {
    /// Builds linear stack of layers into a model sequentially
    StochasticGradientDescent {
        learning_rate: &'a f32,
        momentum: &'a Option<f32>,
    },
}

trait OptimizerInterface<'a> {
    fn create(&'a mut self) -> Result<()>;

    fn get_params<K, V>(&'a self) -> Result<HashMap<K, V>>;

    fn set_params<K, V>(&'a mut self, values: &'a HashMap<K, V>) -> Result<()>;
}

impl<'a> OptimizerInterface<'a> for Optimizer<'a> {
    fn create(&'a mut self) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }

    fn get_params<K, V>(&'a self) -> Result<HashMap<K, V>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }

    fn set_params<K, V>(&'a mut self, values: &'a HashMap<K, V>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
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
    fn create(&'a mut self) -> Result<()>;
    fn add(&'a mut self, layer: &Layer<'a, T>) -> Result<()>;
    fn pop(&'a mut self, layer: &Layer<'a, T>) -> Result<()>;
    fn compile(
        &'a self,
        optimizer: &Optimizer<'a>,
        loss: &'a str,
        metrics: &[&'a str],
        validation_split: &'a f32,
    ) -> Result<Model<T>>;
}

impl<'a, T: num_traits::float::Float> BuildModel<'a, T> for ModelConstructor<'a, T> {
    fn create(&'a mut self) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn add(&'a mut self, layer: &Layer<'a, T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn pop(&'a mut self, layer: &Layer<'a, T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn compile(
        &'a self,
        optimizer: &Optimizer<'a>,
        loss: &'a str,
        metrics: &[&'a str],
        validation_split: &'a f32,
    ) -> Result<Model<T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
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
    fn fit(&'a mut self, inputs: &'a Array2<T>, target: &'a Array2<T>) -> Result<()>;

    fn predict(&'a self, inputs: &'a Array2<T>) -> Result<Array1<T>>;

    fn history(&'a self, key: Option<String>) -> Result<HashMap<u32, HashMap<String, T>>>;

    fn history_plot(&'a self, key: Option<String>) -> Result<()>;

    fn metrics(
        &'a self,
        inputs: &'a Array2<T>,
        target: &'a Array2<T>,
        key: Option<String>,
    ) -> Result<HashMap<String, T>>;
}

impl<'a, T: num_traits::float::Float> UseModel<'a, T> for Model<'a, T> {
    fn fit(&'a mut self, inputs: &'a Array2<T>, target: &'a Array2<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn predict(&'a self, inputs: &'a Array2<T>) -> Result<Array1<T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn history(&'a self, key: Option<String>) -> Result<HashMap<u32, HashMap<String, T>>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }

    fn history_plot(&'a self, key: Option<String>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn metrics(
        &'a self,
        inputs: &'a Array2<T>,
        target: &'a Array2<T>,
        key: Option<String>,
    ) -> Result<HashMap<String, T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
}

impl<'a, T: num_traits::float::Float> BuildModel<'a, T> for Model<'a, T> {
    fn create(&'a mut self) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn add(&'a mut self, layer: &Layer<'a, T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn pop(&'a mut self, layer: &Layer<'a, T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn compile(
        &'a self,
        optimizer: &Optimizer<'a>,
        loss: &'a str,
        metrics: &[&'a str],
        validation_split: &'a f32,
    ) -> Result<Model<T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
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
            activation.calculate_value(&array![[0.0]]).unwrap(),
            &array![[f(0.0)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0, 1.0]]).unwrap(),
            &array![[f(0.0), f(1.0)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0], [1.0]]).unwrap(),
            &array![[f(0.0)], [f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_value(&array![[0.0, 1.0], [2.0, 3.0]])
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
            activation.calculate_value(&array![[0.0]]).unwrap(),
            &array![[f(0.0)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0, 1.0]]).unwrap(),
            &array![[f(0.0), f(1.0)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0], [1.0]]).unwrap(),
            &array![[f(0.0)], [f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_value(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)], [f(2.0), f(3.0)]]
        );
    }

    #[test]
    fn test_relu_activation() {
        fn f(x: f32, alpha: f32, max_value: f32, threshold: f32) -> f32 {
            if x > max_value {
                max_value
            } else if x > threshold {
                x
            } else {
                alpha * (x - threshold)
            }
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
            activation.calculate_value(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0, 1.0]]).unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0], [1.0]]).unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate_value(&array![[0.0, 1.0], [2.0, 3.0]])
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
            activation.calculate_value(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0, 1.0]]).unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation.calculate_value(&array![[0.0], [1.0]]).unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate_value(&array![[0.0, 1.0], [2.0, 3.0]])
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
