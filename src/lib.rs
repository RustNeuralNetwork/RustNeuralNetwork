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
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

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

    #[error("{0}: if given dimensions of a layer are invalid")]
    InvalidLayerDimensions(&'a str),

    #[error("{0}: if weights of given layer is not initialized")]
    WeightsNotInitialized(&'a str),

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
        alpha: Option<T>,
        max_value: Option<T>,
        threshold: Option<T>,
    },

    /// SoftMax Activation function
    SoftMax { axis: usize },
}
/// Functions supported by
/// * rust_neural_network::Activation`
pub trait ActivationInterface<'a, T: num_traits::float::Float> {
    fn calculate_value(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;
    fn calculate_derivative(&'a self, inputs: &'a Array2<T>) -> Result<Array2<T>>;
    fn is_valid(&'a self) -> Result<()>;
}

/// Set of Helper functions to refactor the Trait implementations for better readability

/// Sigmoid function that takes in and returns a generic float type
fn sigmoid<T: num_traits::float::Float>(x: T) -> T {
    T::from(1.0).unwrap() / (T::from(1.0).unwrap() + x.exp())
}

fn relu<T: num_traits::float::Float>(
    x: T,
    alpha: Option<T>,
    max_value: Option<T>,
    threshold: Option<T>,
) -> T {
    let t = match threshold {
        None => T::from(0.0).unwrap(),
        Some(v) => v,
    };
    let a = match alpha {
        None => T::from(0.0).unwrap(),
        Some(v) => v,
    };

    let result = if x > t { x } else { a * (x - t) };

    match max_value {
        None => result,
        Some(m) => {
            if x > m {
                m
            } else {
                result
            }
        }
    }
}

fn relu_derivative<T: num_traits::float::Float>(
    x: T,
    alpha: Option<T>,
    max_value: Option<T>,
    threshold: Option<T>,
) -> T {
    let t = match threshold {
        None => T::from(0.0).unwrap(),
        Some(v) => v,
    };
    let a = match alpha {
        None => T::from(0.0).unwrap(),
        Some(v) => v,
    };
    let result = if x > t {
        T::from(1.0).unwrap()
    } else if x == t {
        (T::from(1.0).unwrap() + a) / T::from(2.0).unwrap()
    } else {
        a
    };
    match max_value {
        None => result,
        Some(m) => {
            if x > m {
                T::from(0.0).unwrap()
            } else if x == m {
                T::from(0.5).unwrap()
            } else {
                result
            }
        }
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

    /// Validates if given activation function is a valid/supported
    fn is_valid(&'a self) -> Result<()> {
        match self {
            Activation::Sigmoid => Ok(()),
            Activation::Tanh => Ok(()),
            Activation::ReLu {
                alpha: _,
                max_value: _,
                threshold: _,
            } => Ok(()),
            Activation::SoftMax { axis: _ } => Ok(()),
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

/// Functions supported by
/// * rust_neural_network::Loss`
pub trait LossInterface<'a, T: num_traits::float::Float> {
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

    fn is_valid(&'a self) -> Result<()>;
}

/// As of now library only supports MSE loss and not entropy.
impl<'a, T: num_traits::float::Float> LossInterface<'a, T> for Loss {
    /// Calculates loss values using multiple functions used in NN base ML models
    ///
    /// # Errors
    ///
    /// * `ModelError::MissingImplementation`
    ///     * if variant `Loss::Entropy` is used
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
    ///
    /// # Errors
    ///
    /// * `ModelError::MissingImplementation`
    ///     * if variant `Loss::Entropy` is used
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
    ///
    /// # Errors
    ///
    /// * `ModelError::MissingImplementation`
    ///     * if variant `Loss::Entropy` is used
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
    ///
    /// # Errors
    ///
    /// * `ModelError::MissingImplementation`
    ///     * if variant `Loss::Entropy` is used
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

    /// Validates if given activation function is a valid/supported
    fn is_valid(&'a self) -> Result<()> {
        match self {
            Loss::MeanSquareError => Ok(()),
            Loss::Entropy => Err(ModelError::MissingImplementation("Entropy")),
        }
    }
}
/// Different types of Layers to construct a Neural Network
#[derive(Clone)]
pub enum Layer<T: num_traits::float::Float> {
    /// Regular densely-connected Neural Network Layer
    Dense {
        activation: Activation<T>,
        input_dim: usize,
        output_dim: usize,
        weights: Option<Array2<T>>,
        loss_values: Option<Array2<T>>,
        prev_layer: Box<Option<Layer<T>>>,
        next_layer: Box<Option<Layer<T>>>,
    },
}
/// Functions supported by
/// * rust_neural_network::Layer`
pub trait ConfigureLayer<'a, T: num_traits::float::Float> {
    fn create(&'a mut self) -> Result<()>;

    fn shape(&'a self) -> Vec<usize>;

    fn get_weights(&'a self) -> Result<Array2<T>>;

    fn set_weights(&'a mut self, weights: Array2<T>) -> Result<()>;

    fn get_layer(&'a self, key: String) -> Result<Box<Option<Layer<T>>>>;

    fn set_layer(
        &'a mut self,
        layer: &'a Option<Layer<T>>,
        key: String,
        reset_weights: bool,
    ) -> Result<()>;

    fn forward_propagate(&'a self, inputs: Array2<T>) -> Result<Array2<T>>;

    fn back_propagate(&'a self, inputs: Array2<T>, targets: Array2<T>) -> Result<()>;
}

impl<'a, T: num_traits::float::Float> ConfigureLayer<'a, T> for Layer<T> {
    /// Validates and creates a Layer to construct a Neural Network
    ///
    /// # Examples
    ///
    /// ```
    /// # use rust_neural_network::Layer;
    /// # use rust_neural_network::Activation;
    /// # use crate::rust_neural_network::ConfigureLayer;
    /// let mut layer: Layer<f32> = Layer::Dense {
    ///     activation: Activation::Sigmoid,
    ///     input_dim:  5,
    ///     output_dim: 1,
    ///     weights: None,
    ///     loss_values: None,
    ///     prev_layer: Box::new(None),
    ///     next_layer: Box::new(None),
    /// };
    /// assert_eq!(layer.create().is_ok(),true)
    /// ```
    /// # Errors
    ///
    /// * `ModelError::InvalidActivationFunction`
    ///     * if `activation` is not implemented/supported
    /// * `ModelError::DimensionMismatch`
    ///     * if `input_dim` incompatible  with `prev_layer.output_dim`
    ///     * if `output_dim` incompatible  with `next_layer.input_dim`
    /// * `ModelError::InvalidLayerDimensions`
    ///     * if `input_dim <= 0`
    ///     * if `output_dim <= 0`
    fn create(&'a mut self) -> Result<()> {
        match self {
            Layer::Dense {
                activation,
                input_dim,
                output_dim,
                weights: _,
                loss_values: _,
                prev_layer,
                next_layer,
            } => {
                let layer_below = prev_layer.to_owned();
                let layer_above = next_layer.to_owned();
                if activation.to_owned().is_valid().is_err() {
                    Err(ModelError::InvalidActivationFunction("Layer::Dense"))
                } else if (layer_below.is_none() || layer_below.unwrap().shape()[1] == *input_dim)
                    && (layer_above.is_none() || layer_above.unwrap().shape()[0] == *output_dim)
                {
                    if *input_dim > 0 || *output_dim > 0 {
                        let new_weights =
                            Array::random((*output_dim, *input_dim), Uniform::new(-0.05, 0.05))
                                .mapv(|x| T::from(x).unwrap());
                        self.set_weights(new_weights)
                    } else {
                        Err(ModelError::InvalidLayerDimensions("Layer::Dense"))
                    }
                } else {
                    Err(ModelError::DimensionMismatch("Layer::Dense"))
                }
            }
        }
    }

    /// Returns the input and output dimensions of the Layer
    ///
    /// # Examples
    ///
    /// ```
    /// # use rust_neural_network::Layer;
    /// # use rust_neural_network::Activation;
    /// # use crate::rust_neural_network::ConfigureLayer;
    /// let mut layer: Layer<f32> = Layer::Dense {
    ///     activation: Activation::Sigmoid,
    ///     input_dim:  5,
    ///     output_dim: 1,
    ///     weights: None,
    ///     loss_values: None,
    ///     prev_layer: Box::new(None),
    ///     next_layer: Box::new(None),
    /// };
    /// assert_eq!(layer.shape(),[5,1].to_vec())
    /// ```
    fn shape(&'a self) -> Vec<usize> {
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
        match self {
            Layer::Dense {
                activation: _,
                input_dim: _,
                output_dim: _,
                weights,
                loss_values: _,
                prev_layer: _,
                next_layer: _,
            } => {
                if let Some(wt) = weights.to_owned() {
                    Ok(wt)
                } else {
                    Err(ModelError::WeightsNotInitialized("Layer::Dense"))
                }
            }
        }
    }

    fn set_weights(&'a mut self, new_weights: Array2<T>) -> Result<()> {
        match self {
            Layer::Dense {
                activation,
                input_dim,
                output_dim,
                weights: _,
                loss_values,
                prev_layer,
                next_layer,
            } => {
                if new_weights.shape()[0] == *output_dim && new_weights.shape()[1] == *input_dim {
                    *self = Layer::Dense {
                        activation: activation.to_owned(),
                        input_dim: input_dim.to_owned(),
                        output_dim: output_dim.to_owned(),
                        weights: Some(new_weights),
                        loss_values: loss_values.to_owned(),
                        prev_layer: prev_layer.to_owned(),
                        next_layer: next_layer.to_owned(),
                    };
                    Ok(())
                } else {
                    Err(ModelError::DimensionMismatch("Layer::Dense"))
                }
            }
        }
    }

    fn get_layer(&'a self, key: String) -> Result<Box<Option<Layer<T>>>> {
        match self {
            Layer::Dense {
                activation: _,
                input_dim: _,
                output_dim: _,
                weights: _,
                loss_values: _,
                prev_layer,
                next_layer,
            } => match key.as_str() {
                "below" => Ok(prev_layer.to_owned()),
                "above" => Ok(next_layer.to_owned()),
                _ => Err(ModelError::ValueNotInRange(
                    "Only allowed key values are [above,below]",
                )),
            },
        }
    }

    fn set_layer(
        &'a mut self,
        layer: &'a Option<Layer<T>>,
        key: String,
        reset_weights: bool,
    ) -> Result<()> {
        match self {
            Layer::Dense {
                activation,
                input_dim,
                output_dim,
                weights,
                loss_values,
                prev_layer,
                next_layer,
            } => {
                if key == "below" {
                    *self = Layer::Dense {
                        activation: activation.to_owned(),
                        input_dim: layer.to_owned().unwrap().shape()[1],
                        output_dim: output_dim.to_owned(),
                        weights: weights.to_owned(),
                        loss_values: loss_values.to_owned(),
                        prev_layer: Box::new(layer.to_owned()),
                        next_layer: next_layer.to_owned(),
                    };
                    if reset_weights {
                        self.create()
                    } else {
                        Ok(())
                    }
                } else if key == "above" {
                    *self = Layer::Dense {
                        activation: activation.to_owned(),
                        input_dim: input_dim.to_owned(),
                        output_dim: layer.to_owned().unwrap().shape()[1],
                        weights: weights.to_owned(),
                        loss_values: loss_values.to_owned(),
                        prev_layer: prev_layer.to_owned(),
                        next_layer: Box::new(layer.to_owned()),
                    };
                    if reset_weights {
                        self.create()
                    } else {
                        Ok(())
                    }
                } else {
                    Err(ModelError::ValueNotInRange(
                        "Only allowed key values are [above,below]",
                    ))
                }
            }
        }
    }

    fn forward_propagate(&'a self, _inputs: Array2<T>) -> Result<Array2<T>> {
        Err(ModelError::MissingImplementation(
            "ConfigureLayer::forward_propagate",
        ))
    }

    fn back_propagate(&'a self, _inputs: Array2<T>, _targets: Array2<T>) -> Result<()> {
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

    fn set_params<K, V>(&'a mut self, _values: &'a HashMap<K, V>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
}

/// Different types of NN Model Constructors
#[derive(Clone)]
pub enum ModelConstructor<'a, T: num_traits::float::Float> {
    /// Builds linear stack of layers into a model sequentially
    Sequential {
        name: &'a str,
        layers: Vec<Layer<T>>,
    },
}

trait BuildModel<'a, T: num_traits::float::Float> {
    fn create(&'a mut self) -> Result<()>;
    fn add(&'a mut self, layer: &Layer<T>) -> Result<()>;
    fn pop(&'a mut self, layer: &Layer<T>) -> Result<()>;
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
    fn add(&'a mut self, _layer: &Layer<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn pop(&'a mut self, _layer: &Layer<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn compile(
        &'a self,
        _optimizer: &Optimizer<'a>,
        _loss: &'a str,
        _metrics: &[&'a str],
        _validation_split: &'a f32,
    ) -> Result<Model<T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
}

/// Groups a linear stack of layers into a Model
#[derive(Clone)]
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
    fn fit(&'a mut self, _inputs: &'a Array2<T>, _target: &'a Array2<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn predict(&'a self, _inputs: &'a Array2<T>) -> Result<Array1<T>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn history(&'a self, _key: Option<String>) -> Result<HashMap<u32, HashMap<String, T>>> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }

    fn history_plot(&'a self, _key: Option<String>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn metrics(
        &'a self,
        _inputs: &'a Array2<T>,
        _target: &'a Array2<T>,
        _key: Option<String>,
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
    fn add(&'a mut self, _layer: &Layer<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn pop(&'a mut self, _layer: &Layer<T>) -> Result<()> {
        Err(ModelError::MissingImplementation(
            "OptimizerInterface::create",
        ))
    }
    fn compile(
        &'a self,
        _optimizer: &Optimizer<'a>,
        _loss: &'a str,
        _metrics: &[&'a str],
        _validation_split: &'a f32,
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
        fn f(x: f32, alpha: Option<f32>, max_value: Option<f32>, threshold: Option<f32>) -> f32 {
            let t = match threshold {
                None => 0.0,
                Some(v) => v,
            };
            let a = match alpha {
                None => 0.0,
                Some(v) => v,
            };

            let result = if x > t { x } else { a * (x - t) };

            match max_value {
                None => result,
                Some(m) => {
                    if x > m {
                        m
                    } else {
                        result
                    }
                }
            }
        }

        let mut alpha = Some(0.0);
        let mut max_value = None;
        let mut threshold = Some(0.0);
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
        alpha = Some(0.001);
        max_value = Some(1.0);
        threshold = Some(0.0);
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

    #[test]
    fn test_sigmoid_derivative() {
        fn f(x: f32) -> f32 {
            let y = 1.0 / (1.0 + x.exp());
            y * (1.0 - y)
        }
        let activation = Activation::Sigmoid;
        assert_eq!(
            activation.calculate_derivative(&array![[0.0]]).unwrap(),
            &array![[f(0.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0], [1.0]])
                .unwrap(),
            &array![[f(0.0)], [f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)], [f(2.0), f(3.0)]]
        );
    }

    #[test]
    fn test_tanh_derivative() {
        fn f(x: f32) -> f32 {
            1.0 - x.tanh().powf(2.0)
        }

        let activation = Activation::Tanh;
        assert_eq!(
            activation.calculate_derivative(&array![[0.0]]).unwrap(),
            &array![[f(0.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0], [1.0]])
                .unwrap(),
            &array![[f(0.0)], [f(1.0)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0], [2.0, 3.0]])
                .unwrap(),
            &array![[f(0.0), f(1.0)], [f(2.0), f(3.0)]]
        );
    }

    #[test]
    fn test_relu_derivative() {
        fn f(x: f32, alpha: Option<f32>, max_value: Option<f32>, threshold: Option<f32>) -> f32 {
            let t = match threshold {
                None => 0.0,
                Some(v) => v,
            };
            let a = match alpha {
                None => 0.0,
                Some(v) => v,
            };
            let result = if x > t {
                1.0
            } else if x == t {
                (1.0 + a) / 2.0
            } else {
                a
            };
            match max_value {
                None => result,
                Some(m) => {
                    if x > m {
                        0.0
                    } else if x == m {
                        0.5
                    } else {
                        result
                    }
                }
            }
        }

        let mut alpha = Some(0.0);
        let mut max_value = None;
        let mut threshold = Some(0.0);
        let mut activation = Activation::ReLu {
            alpha,
            max_value,
            threshold,
        };
        assert_eq!(
            activation.calculate_derivative(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0]])
                .unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0], [1.0]])
                .unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0], [2.0, 3.0]])
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
        alpha = Some(0.001);
        max_value = Some(1.0);
        threshold = Some(0.0);
        activation = Activation::ReLu {
            alpha,
            max_value,
            threshold,
        };
        assert_eq!(
            activation.calculate_derivative(&array![[0.0]]).unwrap(),
            &array![[f(0.0, alpha, max_value, threshold)]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0]])
                .unwrap(),
            &array![[
                f(0.0, alpha, max_value, threshold),
                f(1.0, alpha, max_value, threshold)
            ]]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0], [1.0]])
                .unwrap(),
            &array![
                [f(0.0, alpha, max_value, threshold)],
                [f(1.0, alpha, max_value, threshold)]
            ]
        );
        assert_eq!(
            activation
                .calculate_derivative(&array![[0.0, 1.0], [2.0, 3.0]])
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
