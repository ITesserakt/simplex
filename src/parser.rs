use std::{fmt::Debug, str::FromStr};

use nom::{
    branch::alt,
    bytes::{
        complete::{tag, tag_no_case},
    },
    character::complete::char,
    character::complete::{multispace0, one_of, line_ending},
    combinator::{opt, recognize},
    error::{context, ContextError, ParseError},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, terminated, preceded},
    IResult, Parser,
};
use num::{One, Rational64};

#[derive(PartialEq, Debug, Clone)]
pub enum Goal {
    Maximize,
    Minimize,
}

#[derive(PartialEq, Debug)]
pub enum Relation {
    Equal,
    Less,
    Greater,
}

#[derive(Debug, PartialEq)]
pub struct Term {
    pub coef: Rational64,
    pub index: u64,
}

#[derive(Debug, PartialEq)]
pub struct TargetFn {
    pub goal: Goal,
    pub terms: Vec<Term>,
    pub value: Rational64,
}

#[derive(Debug, PartialEq)]
pub struct Restriction {
    pub relation: Relation,
    pub terms: Vec<Term>,
    pub value: Rational64,
}

#[derive(Debug, PartialEq)]
pub struct Task {
    pub restrictions: Vec<Restriction>,
    pub target_fn: TargetFn,
}

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading and
/// trailing whitespace, returning the output of `inner`.
fn ws<'a, F, O, E>(inner: F) -> impl Parser<&'a str, O, E>
where
    F: Parser<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(multispace0, inner, multispace0)
}

fn decimal<'a, E>(input: &'a str) -> IResult<&'a str, u64, E>
where
    E: ParseError<&'a str>,
{
    recognize(many1(terminated(one_of("0123456789"), many0(char('_')))))
        .parse(input)
        .map(|x| (x.0, x.1.parse().unwrap()))
}

fn coefficient<'a, E>() -> impl Parser<&'a str, Rational64, E>
where
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    context("coefficient", move |s| {
        let (s, sign) = opt(one_of("+-")).parse(s)?;
        let (s, whole) = decimal.parse(s)?;
        let (s, trunc) = opt(|s| {
            let (s, _) = tag(".").parse(s)?;
            opt(decimal).parse(s)
        })
        .parse(s)?;

        let whole = whole as i64;
        let trunc = trunc.flatten().unwrap_or(0);
        let (power, trunc) = if trunc == 0 {
            (1, 0)
        } else {
            (10_i64.pow(trunc.ilog10() + 1), trunc as i64)
        };
        let number = Rational64::new_raw(whole, 1) + Rational64::new(trunc, power);

        Ok((
            s,
            if let Some('-') = sign {
                -number
            } else {
                number
            },
        ))
    })
    .or(char('-').map(|_| (-1).into()))
}

/// <0..9>+( *'*' *)?x<0..9>+
fn term<'a, E>() -> impl Parser<&'a str, Term, E>
where
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    context("term", move |s| {
        let (s, coef) = opt(coefficient()).parse(s)?;
        let (s, _) = opt(ws(tag("*"))).parse(s)?;
        let (s, _) = tag_no_case("x").parse(s)?;
        let (s, index) = decimal(s)?;

        Ok((
            s,
            Term {
                coef: coef.unwrap_or(Rational64::one()),
                index,
            },
        ))
    })
}

/// 'z' *'=' *([inner] *'+')+ *-> *('max'|'min')
fn target_fn<'a, E>() -> impl Parser<&'a str, TargetFn, E>
where
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    context("target_fn", |s| {
        let (s, _) = tag_no_case("z").parse(s)?;
        let (s, _) = ws(tag("=")).parse(s)?;
        let (s, terms) = separated_list1(ws(char('+')), term()).parse(s)?;
        let (s, _) = ws(tag("->")).parse(s)?;
        let (s, goal) = alt((tag_no_case("max"), tag_no_case("min"))).parse(s)?;

        Ok((
            s,
            TargetFn {
                goal: if goal.to_lowercase() == "max" {
                    Goal::Maximize
                } else {
                    Goal::Minimize
                },
                terms,
                value: Default::default(),
            },
        ))
    })
}

/// '=='|'<='|'>='
fn relation<'a, E>() -> impl Parser<&'a str, Relation, E>
where
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    context("relation", |s| {
        let (rest, choise) = tag("==").or(tag("<=")).or(tag(">=")).parse(s)?;

        Ok((
            rest,
            match choise {
                "==" => Relation::Equal,
                "<=" => Relation::Less,
                ">=" => Relation::Greater,
                _ => unreachable!(),
            },
        ))
    })
}

/// ([term] *'+') *[relation] *[value]
fn restriction<'a, E>() -> impl Parser<&'a str, Restriction, E>
where
    E: ParseError<&'a str> + ContextError<&'a str>,
{
    context("restriction", |s| {
        let (s, terms) = separated_list1(ws(char('+')), term()).parse(s)?;
        let (s, relation) = ws(relation()).parse(s)?;
        let (s, value) = preceded(multispace0, coefficient()).parse(s)?;

        Ok((
            s,
            Restriction {
                relation,
                terms,
                value,
            },
        ))
    })
}

impl Task {
    fn parse<'a, E>() -> impl Parser<&'a str, Task, E>
    where
        E: ParseError<&'a str> + ContextError<&'a str>,
    {
        context("task", |s| {
            let (s, restrictions) = separated_list1(line_ending, restriction()).parse(s)?;
            let (s, _) = line_ending(s)?;
            let (s, target_fn) = target_fn().parse(s)?;

            Ok((
                s,
                Self {
                    restrictions,
                    target_fn,
                },
            ))
        })
    }
}

impl FromStr for Task {
    type Err = nom::Err<nom::error::VerboseError<String>>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Task::parse::<nom::error::VerboseError<&str>>()
            .parse(s)
            .map(|x| x.1)
            .map_err(|x| {
                x.map(|y| nom::error::VerboseError {
                    errors: y
                        .errors
                        .into_iter()
                        .map(|z| (z.0.to_owned(), z.1))
                        .collect(),
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use nom::Parser;
    use num::Rational64;
    use rstest::rstest;

    use crate::parser::{
        coefficient, relation, restriction, target_fn, Goal, Relation, Restriction, TargetFn, Term,
    };

    #[rstest]
    #[case("5.2", 5.2)]
    #[case("-555.111", -555.111)]
    #[case("5.", 5.0)]
    #[case("5", 5.0)]
    fn test_coefficient(#[case] num_str: &str, #[case] number: f64) {
        assert_eq!(
            coefficient::<nom::error::Error<&str>>().parse(num_str),
            Ok(("", Rational64::approximate_float(number).unwrap()))
        );
    }

    #[rstest]
    fn test_target_fn() {
        assert_eq!(
            target_fn::<nom::error::Error<&str>>().parse("z = 2x1 -> min"),
            Ok((
                "",
                TargetFn {
                    goal: Goal::Minimize,
                    terms: vec![Term {
                        coef: 2.into(),
                        index: 1
                    }],
                    value: Default::default()
                }
            ))
        );

        assert_eq!(
            target_fn::<nom::error::Error<&str>>().parse("z =  5 * x2  + -x4  -> max"),
            Ok((
                "",
                TargetFn {
                    goal: Goal::Maximize,
                    terms: vec![
                        Term {
                            coef: 5.into(),
                            index: 2
                        },
                        Term {
                            coef: (-1).into(),
                            index: 4
                        }
                    ],
                    value: Default::default()
                }
            ))
        );

        assert_eq!(
            target_fn::<nom::error::Error<&str>>().parse("z=x0    ->    min"),
            Ok((
                "",
                TargetFn {
                    goal: Goal::Minimize,
                    terms: vec![Term {
                        coef: 1.into(),
                        index: 0
                    }],
                    value: Default::default()
                }
            ))
        );

        assert!(target_fn::<nom::error::Error<&str>>()
            .parse("z = min")
            .is_err());
    }

    #[rstest]
    #[case("x1 + 2x2 == 3", Restriction {
        relation: Relation::Equal,
        terms: vec![Term {
            coef: 1.into(),
            index: 1
        }, Term {
            coef: 2.into(),
            index: 2
        }],
        value: 3.into()
    })]
    fn test_restriction(#[case] input: &str, #[case] res: Restriction) {
        assert_eq!(
            restriction::<nom::error::Error<&str>>().parse(input),
            Ok(("", res))
        )
    }

    #[rstest]
    #[case("==", Relation::Equal)]
    #[case("<=", Relation::Less)]
    #[case(">=", Relation::Greater)]
    #[should_panic]
    #[case("NaN", Relation::Equal)]
    fn test_relation(#[case] rel_str: &str, #[case] rel: Relation) {
        assert_eq!(
            relation::<nom::error::Error<&str>>().parse(rel_str),
            Ok(("", rel))
        );
    }
}
