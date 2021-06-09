extern crate rust_neural_network;
use csv;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
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
            // start with model constructor
            // let model: Model;
            // let mut layer: Layer<f32> = Layer::Dense {
            //     activation: Activation::Sigmoid,
            //     input_dim:  5,
            //     output_dim: 1,
            //     weights: None,
            //     loss_values: None,
            //     prev_layer: Box::new(None),
            //     next_layer: Box::new(None),
            // };
        }
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    }
}
