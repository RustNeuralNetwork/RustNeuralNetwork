extern crate rust_neural_network;
use csv;
use ndarray::s;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use rust_neural_network::Activation;
use rust_neural_network::BuildModel;
use rust_neural_network::ConfigureLayer;
use rust_neural_network::Layer;
use rust_neural_network::Loss;
use rust_neural_network::ModelConstructor;
use rust_neural_network::Optimizer;
use rust_neural_network::UseModel;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;

// resource: https://docs.rs/csv/1.1.6/csv/tutorial/index.html#reading-csv

fn read_csv(
    test_path: OsString,
    train_path: OsString,
) -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let test_file = File::open(test_path)?;
    let train_file = File::open(train_path)?;
    let mut test_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(test_file);
    let mut train_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(train_file);
    let test_arr: Array2<f32> = test_reader.deserialize_array2((10000, 28 * 28 + 1))?;
    let train_arr: Array2<f32> = train_reader.deserialize_array2((60000, 28 * 28 + 1))?;
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
    let (train_data, test_data) = read_csv(test_path, train_path).unwrap();
    //let mut train_x = train_data.slice(s![.., 1..]);
    //let mut train_target = train_data.slice(s![.., 0]);
    //let mut test_x = test_data.slice(s![.., 1..]);
    //let mut test_target = test_data.slice(s![.., 0]);
    let mut model_constructor: ModelConstructor<'_, f32> = ModelConstructor::Sequential {
        name: "model",
        layers: None,
        input_dim: 28 * 28,
        output_dim: Some(10),
    };
    model_constructor.create();

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

    model_constructor.add(&mut layer1);
    model_constructor.add(&mut layer2);
    model_constructor.add(&mut layer3);

    let model = model_constructor
        .compile(
            &Optimizer::StochasticGradientDescent {
                learning_rate: &0.5,
                momentum: &None,
            },
            &Loss::MeanSquareError,
            &[&"loss"],
            &0.5,
        )
        .unwrap();

    //model.fit(&train_x.to_owned(), &train_target.to_owned(), &50);

    //let mut predictions = model.predict(&test_x.to_owned()).unwrap();

    //assert_eq!(test_target, predictions);
}
