#![feature(slice_flatten)]
#![feature(drain_filter)]

use std::{
    env::args,
    fs::read_to_string,
};

use num::rational::Ratio;

use crate::{task::SimplexTask, simplex::SimplexSolver, tax_numbers::Tax};

mod errors;
mod simplex;
mod task;
mod tax_numbers;

fn main() {
    let input_path = args().nth(1).expect("No input file path passed");
    let input = read_to_string(input_path).unwrap();

    let task: SimplexTask<Tax<Ratio<i32>>> = input.parse().expect("Cannot parse given input");
    let solver: SimplexSolver<_> = task.canonize().into();
    let solution = solver.solve().expect("Cannot get solution");

    println!("{solution}");
}
