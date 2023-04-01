use std::fmt::Debug;

#[allow(dead_code)]
#[derive(Debug)]
pub enum SimplexMethodError {
    NoLimit,
    NoSolutions,
}


#[derive(Debug)]
pub enum SimplexParseErr {
    UnexpectedRelation,
    EndOfInput,
    NotANumber,
    NoTarget,
    Composite(Box<SimplexParseErr>, Box<SimplexParseErr>)
}
