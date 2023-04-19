#![feature(slice_flatten)]
#![feature(drain_filter)]

use std::{
    env::args,
    fs::read_to_string,
};

use num::Rational64;

use crate::{task::{SimplexTask, Simple, Taxes, DoublePhase}, simplex::SimplexSolver, tax_numbers::Tax, parser::Task};

mod errors;
mod simplex;
mod task;
mod tax_numbers;
mod parser;

fn main() {
    let input_path = args().nth(1).unwrap_or("input.txt".to_owned());
    let input = read_to_string(input_path).unwrap();

    let task: Task = input.parse().expect("Cannot parse given input");
    let method = task.method;
    let task: SimplexTask<Tax<Rational64>> = task.into();
    let solver: SimplexSolver<Tax<Rational64>> = match method {
        parser::Method::Simple => task.canonize::<Simple>().into(),
        parser::Method::Taxes => task.canonize::<Taxes>().into(),
        parser::Method::SecondPhase => task.canonize::<DoublePhase>().into(),
    };
    let solution = solver.solve().expect("Cannot get solution");

    println!("{solution}");
}
