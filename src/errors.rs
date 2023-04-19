use std::fmt::Debug;

#[allow(dead_code)]
#[derive(Debug)]
pub enum SimplexMethodError {
    NoLimit,
    NoSolutions,
}
