extern crate rust_neural_network;
use csv;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use rust_neural_network::Activation;
use rust_neural_network::BuildModel;
use rust_neural_network::ConfigureLayer;
use rust_neural_network::Layer;
use rust_neural_network::Loss;
use rust_neural_network::ModelConstructor;
use rust_neural_network::Optimizer;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::process;

// resource: https://docs.rs/csv/1.1.6/csv/tutorial/index.html#reading-csv

fn read_csv(
    test_path: OsString,
    train_path: OsString,
) -> Result<(Array2<u64>, Array2<u64>), Box<dyn Error>> {
    let test_file = File::open(test_path)?;
    let train_file = File::open(train_path)?;
    let mut test_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(test_file);
    let mut train_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(train_file);
    let test_arr: Array2<u64> = test_reader.deserialize_array2((10000, 28 * 28 + 1))?;
    let train_arr: Array2<u64> = train_reader.deserialize_array2((60000, 28 * 28 + 1))?;
    Ok((test_arr, train_arr))
}

fn get_args(n: usize) -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(n) {
        None => Err(From::from("Expected argument, but got none")),
        Some(path) => Ok(path),
    }
}

fn main() {
    let test_path = get_args(1).unwrap();
    let train_path = get_args(2).unwrap();
    match read_csv(test_path, train_path) {
        Ok((train_data, test_data)) => {
            println!("Success");
            let model_constructor: ModelConstructor<'_, f32> = ModelConstructor::Sequential {
                name: "model",
                layers: None,
                input_dim: 28 * 28,
                output_dim: Some(10),
            };
            match model_constructor.create() {
                Ok(()) => {
                    let mut layer1: Layer<f32> = Layer::Dense {
                        activation: Activation::Sigmoid,
                        input_dim: 28 * 28,
                        output_dim: 100,
                        weights: None,
                        loss_values: None,
                        prev_layer: Box::new(None),
                        next_layer: Box::new(None),
                    };
                    let mut layer2: Layer<f32> = Layer::Dense {
                        activation: Activation::Sigmoid,
                        input_dim: 100,
                        output_dim: 100,
                        weights: None,
                        loss_values: None,
                        prev_layer: Box::new(None),
                        next_layer: Box::new(None),
                    };
                    let mut layer3: Layer<f32> = Layer::Dense {
                        activation: Activation::Sigmoid,
                        input_dim: 100,
                        output_dim: 10,
                        weights: None,
                        loss_values: None,
                        prev_layer: Box::new(None),
                        next_layer: Box::new(None),
                    };
                    layer1.set_layer(&Some(layer2), "above".to_string(), false);
                    layer2.set_layer(&Some(layer1), "below".to_string(), false);
                    layer2.set_layer(&Some(layer2), "above".to_string(), false);
                    layer3.set_layer(&Some(layer2), "below".to_string(), false);
                    model_constructor.add(&layer1);
                    model_constructor.add(&layer2);
                    model_constructor.add(&layer3);
                    match model_constructor.compile(
                        Optimizer::StochasticGradientDescent {
                            learning_rate: &0.5,
                            momentum: &None,
                        },
                        Loss::MeanSquareError,
                        &[&"loss"],
                        validation_split: &'a T,
                    ) {
                        Ok(model) => {}
                        Err(err) => {
                            println!("{}", err);
                            process::exit(1);
                        }
                    }
                }
                Err(err) => {
                    println!("{}", err);
                    process::exit(1);
                }
            };
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    }
}
