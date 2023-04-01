use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    ops::Not,
    str::FromStr,
    vec,
};

use ndarray::{aview0, Array1, Array2, Axis};
use num::{traits::NumAssign, Num};
use regex::Regex;

use crate::{
    errors::SimplexParseErr,
    simplex::{Aim, SimplexSolver},
    tax_numbers::Tax,
};

#[derive(Debug)]
struct Index(usize);

impl Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
enum SimplexRelation {
    Less,
    Eq,
    Greater,
}

#[derive(Debug)]
struct SimplexTerm<F: Debug>(F, Index);

#[derive(Debug)]
struct SimplexFreeTerm<F: Debug>(F);

#[derive(Debug)]
struct SimplexRestriction<F: Debug> {
    terms: Vec<SimplexTerm<F>>,
    relation: SimplexRelation,
    free: SimplexFreeTerm<F>,
}

#[derive(Debug)]
struct SimplexTarget<F: Debug> {
    terms: Vec<SimplexTerm<F>>,
    free: SimplexFreeTerm<F>,
}

#[derive(Debug)]
pub struct SimplexTask<F: Debug> {
    restrictions: Vec<SimplexRestriction<F>>,
    target_fn: SimplexTarget<F>,
}

struct SimplaxTaskParts<F: Debug> {
    a: Array2<F>,
    b: Array1<F>,
    z: Array1<F>,
}

#[derive(Debug)]
pub struct CanonicSimplexTask<F: Debug> {
    task: SimplexTask<F>,
    max_index: usize,
}

impl FromStr for SimplexRelation {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "<=" => Ok(SimplexRelation::Less),
            ">=" => Ok(SimplexRelation::Greater),
            "==" => Ok(SimplexRelation::Eq),
            _ => Err(SimplexParseErr::UnexpectedRelation),
        }
    }
}

impl<F: Debug + Num> Not for SimplexTerm<F> {
    type Output = SimplexTerm<F>;

    fn not(self) -> Self::Output {
        Self(self.0 * (F::zero() - F::one()), self.1)
    }
}

impl<F: Debug + Num> Not for SimplexFreeTerm<F> {
    type Output = SimplexFreeTerm<F>;

    fn not(self) -> Self::Output {
        SimplexFreeTerm(self.0 * (F::zero() - F::one()))
    }
}

impl<F: FromStr + Debug + Num> FromStr for SimplexTerm<F> {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parse_index = |index: &str| {
            index
                .trim()
                .parse()
                .map_err(|_| SimplexParseErr::NotANumber)
                .map(Index)
        };

        let parts = s.split_once('x');

        match parts {
            Some(("-", index)) => Ok(SimplexTerm(F::zero() - F::one(), parse_index(index)?)),
            Some(("+" | "", index)) => Ok(SimplexTerm(F::one(), parse_index(index)?)),
            None => Ok(SimplexTerm(F::one(), parse_index(s)?)),
            Some((coefficient, index)) => Ok(SimplexTerm(
                coefficient
                    .trim()
                    .parse()
                    .map_err(|_| SimplexParseErr::NotANumber)?,
                parse_index(index)?,
            )),
        }
    }
}

impl<F: FromStr + Debug + Num> FromStr for SimplexFreeTerm<F> {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SimplexFreeTerm(
            s.trim().parse().map_err(|_| SimplexParseErr::NotANumber)?,
        ))
    }
}

fn exract_terms<H, T: Debug>(s: &str, parse_handle: H) -> Result<Vec<T>, SimplexParseErr>
where
    H: Fn(&str, bool) -> Result<T, SimplexParseErr>,
{
    Ok(s.split_inclusive(['-', '+'])
        .fold(Ok((vec![], false)), |acc, next| {
            let Ok((mut acc, is_minus)) = acc else { return acc; };

            let minus = next.ends_with('-');
            let next = next.trim_end_matches(['-', '+']);
            let term = parse_handle(next.trim(), is_minus)?;

            acc.push(term);

            Ok((acc, minus))
        })?
        .0)
}

impl<F: FromStr + Debug + Num> FromStr for SimplexRestriction<F> {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let pattern = "<= | == | >=";
        let regex = Regex::new(pattern).unwrap();
        let relation = regex.find(s).ok_or(SimplexParseErr::EndOfInput)?;
        let [terms, free]: [&str; 2] = regex.splitn(s, 2).collect::<Vec<_>>().try_into().unwrap();

        Ok(SimplexRestriction {
            terms: exract_terms(terms, |s, is_minus| {
                s.parse::<SimplexTerm<F>>()
                    .map(|x| if is_minus { !x } else { x })
            })?,
            relation: relation.as_str().trim().parse()?,
            free: free.trim().parse()?,
        })
    }
}

impl<F: FromStr + Debug + Num> FromStr for SimplexTarget<F> {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut vec = exract_terms(s, |s, is_minus| match s.parse::<SimplexTerm<F>>() {
            Ok(term) => Ok(Ok(if is_minus { !term } else { term })),
            Err(_) => match s.parse::<SimplexFreeTerm<F>>() {
                Ok(free) => Ok(Err(if is_minus { !free } else { free })),
                Err(e) => Err(e),
            },
        })?;

        let free = vec
            .drain_filter(|x| x.is_err())
            .map(|x| x.unwrap_err().0)
            .fold(F::zero(), |acc, next| acc + next);
        let vec: Vec<_> = vec.into_iter().map(|x| x.unwrap()).collect();

        if vec.is_empty() {
            return Err(SimplexParseErr::NoTarget);
        }

        Ok(SimplexTarget {
            terms: vec,
            free: SimplexFreeTerm(free),
        })
    }
}

impl<F: FromStr + Debug + Num> FromStr for SimplexTask<F> {
    type Err = SimplexParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut restrictions = vec![];
        let mut target = None;

        for x in s.lines() {
            match x.parse() {
                Ok(restriction) => restrictions.push(restriction),
                Err(e1) => match x.parse() {
                    Ok(tg) => target = Some(tg),
                    Err(e2) => return Err(SimplexParseErr::Composite(Box::new(e1), Box::new(e2))),
                },
            }
        }

        Ok(SimplexTask {
            restrictions,
            target_fn: target.ok_or(SimplexParseErr::NoTarget)?,
        })
    }
}

impl<F: Debug + Num + NumAssign + PartialOrd> SimplexTask<F> {
    pub fn canonize(mut self) -> CanonicSimplexTask<F> {
        let mut max_index = self
            .restrictions
            .iter()
            .flat_map(|x| &x.terms)
            .max_by_key(|x| x.1 .0)
            .unwrap()
            .1
             .0;

        for restriction in &mut self.restrictions {
            match restriction.relation {
                SimplexRelation::Less => {
                    restriction
                        .terms
                        .push(SimplexTerm(F::one(), Index(max_index + 1)));
                    max_index += 1;
                }
                SimplexRelation::Eq => (),
                SimplexRelation::Greater => {
                    restriction
                        .terms
                        .push(!SimplexTerm(F::one(), Index(max_index + 1)));
                    max_index += 1;
                }
            }

            restriction.relation = SimplexRelation::Eq;

            if restriction.free.0 < F::zero() {
                restriction
                    .terms
                    .iter_mut()
                    .for_each(|x| x.0 *= F::zero() - F::one());
                restriction.free.0 *= F::zero() - F::one();
            }
        }
        CanonicSimplexTask {
            task: self,
            max_index,
        }
    }
}

#[cfg(not(feature = "taxes"))]
impl<F: Display + Num + Clone + Debug> Into<SimplexSolver<F>> for CanonicSimplexTask<F> {
    fn into(self) -> SimplexSolver<F> {
        let mut parts = self.into_a_b_z();
        // parts.add_taxes();
        // parts.add_basis();
        parts.invert_z();
        let contents = parts.into_contents();

        SimplexSolver::from_contents(contents, Aim::Maximize)
    }
}

#[cfg(feature = "taxes")]
impl<F: Display + Num + Clone + Debug> Into<SimplexSolver<Tax<F>>> for CanonicSimplexTask<Tax<F>> {
    fn into(self) -> SimplexSolver<Tax<F>> {
        let mut parts = self.into_a_b_z();
        parts.add_taxes();
        parts.add_basis();
        parts.invert_z();
        let contents = parts.into_contents();

        SimplexSolver::from_contents(contents, Aim::Maximize)
    }
}

impl<T: std::fmt::Debug + Num + Clone> CanonicSimplexTask<T> {
    fn into_a_b_z(self) -> SimplaxTaskParts<T> {
        let restrictions_len = self.task.restrictions.len();

        let mut a_hash_map = self
            .task
            .restrictions
            .iter()
            .map(|x| {
                x.terms
                    .iter()
                    .map(|y| (y.1 .0 - 1, y.0.clone()))
                    .collect::<HashMap<_, _>>()
            })
            .enumerate()
            .collect::<HashMap<_, _>>();

        let mut z_hash_map = self
            .task
            .target_fn
            .terms
            .into_iter()
            .map(|x| (x.1 .0 - 1, x.0))
            .collect::<HashMap<_, _>>();

        let a = Array2::from_shape_fn((restrictions_len, self.max_index), |(i, j)| {
            a_hash_map
                .entry(i)
                .or_insert(HashMap::new())
                .entry(j)
                .or_insert(T::zero())
                .clone()
        });
        let b = Array1::from_shape_vec(
            restrictions_len,
            self.task
                .restrictions
                .into_iter()
                .map(|x| x.free.0)
                .collect(),
        )
        .unwrap();
        let mut z = Array1::from_shape_fn(self.max_index, |i| {
            z_hash_map.entry(i).or_insert(T::zero()).clone()
        });
        z.push(Axis(0), aview0(&self.task.target_fn.free.0))
            .unwrap();

        SimplaxTaskParts {
            a: a,
            b: b,
            z: z,
        }
    }
}

impl<T: Debug + std::fmt::Display + num::Num + std::clone::Clone> SimplaxTaskParts<Tax<T>> {
    fn add_taxes(&mut self)
    where
        T: Num + Clone + Display,
    {
        let mut taxed = self.a.sum_axis(Axis(0)).mapv(|x| x.into_tax());
        taxed.push(Axis(0), aview0(&self.b.sum().into_tax())).unwrap();

        self.z.zip_mut_with(&taxed, |x, y| *x = x.clone() + y.clone())
    }
}

impl<T: Debug> SimplaxTaskParts<T> {
    fn add_basis(&mut self) where T: Clone + Num {
        let max_index = self.z.len() - 1;
        let restrictions_len = self.a.len_of(Axis(0));
        self.a.append(Axis(1), Array2::eye(restrictions_len).view()).unwrap();
        self.z.append(Axis(0), Array1::from_elem(restrictions_len, T::zero()).view()).unwrap();
        self.z.swap(max_index, max_index + restrictions_len);
    }

    fn invert_z(&mut self) where T: Num + Clone {
        self.z.map_inplace(|x| *x = x.clone() * (T::zero() - T::one()));
    }

    fn into_contents(mut self) -> Array2<T> where T: Clone {
        self.a.push_column(self.b.view()).unwrap();
        self.a.push_row(self.z.view()).unwrap();

        self.a
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::str::FromStr;

    use num::rational::Ratio;

    use super::SimplexFreeTerm;
    use super::SimplexRelation;
    use super::SimplexRestriction;
    use super::SimplexTarget;
    use super::SimplexTask;
    use super::SimplexTerm;

    fn test_multiple<T: FromStr + Debug>(items: &[(&str, bool)])
    where
        <T as FromStr>::Err: Debug,
    {
        for (item, pass) in items {
            eprintln!("Assert `{item}`:");
            assert!(
                item.parse::<T>().is_ok() == *pass,
                "`{item}` == `{:?}`",
                item.parse::<T>()
            )
        }
    }

    #[test]
    fn test_relation() {
        test_multiple::<SimplexRelation>(&[
            ("<=", true),
            ("==", true),
            (">", false),
            ("", false),
            (">=", true),
        ]);
    }

    #[test]
    fn test_term() {
        test_multiple::<SimplexTerm<f64>>(&[
            ("", false),
            ("x0", true),
            ("x10", true),
            ("-1x4", true),
            ("-x1", true),
            ("1x", false),
            ("x", false),
            ("1.4x1", true),
            ("0.5x2", true),
            ("x1 ", true),
        ]);
        test_multiple::<SimplexTerm<Ratio<i32>>>(&[
            ("", false),
            ("x0", true),
            ("x10", true),
            ("-1x4", true),
            ("-x1", true),
            ("1x", false),
            ("x", false),
            ("14/10x1", true),
            ("1/2x2", true),
            ("0.3x1", false),
        ]);
    }

    #[test]
    fn test_free() {
        test_multiple::<SimplexFreeTerm<f64>>(&[
            ("", false),
            ("1", true),
            ("a", false),
            ("45", true),
            ("     3  ", true),
        ]);
    }

    #[test]
    fn test_restriction() {
        test_multiple::<SimplexRestriction<f64>>(&[
            ("x1 + 2x2 + x3 <= 3", true),
            ("x1 == 4", true),
            ("", false),
            ("x2 == x1", false),
            ("x2 + ", false),
            ("x1 + x2 >= 5", true),
        ]);
    }

    #[test]
    fn test_target() {
        test_multiple::<SimplexTarget<f64>>(&[
            ("x1", true),
            ("x1 + x2", true),
            ("17x1 + 3x2", true),
            ("17x1 == 5", false),
            ("x1 + 5", true),
            ("", false),
        ])
    }

    #[test]
    fn test_task() {
        test_multiple::<SimplexTask<f64>>(&[("x1 == 3\nx2 <= 3\nx1 + x2", true)])
    }
}
