use std::{
    clone::Clone,
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use ndarray::{aview0, Array1, Array2, Axis};
use num::{traits::NumAssign, Num, Rational64, Zero};

use crate::tax_numbers::Tax;
use crate::{
    parser::{Goal, Relation, Task},
    simplex::SimplexSolver,
};

#[derive(Debug)]
struct SimplexTerm<F: Debug> {
    coef: F,
    index: u64,
}

#[derive(Debug)]
struct SimplexRestriction<F: Debug> {
    terms: Vec<SimplexTerm<F>>,
    relation: Relation,
    free: F,
}

#[derive(Debug)]
struct SimplexTarget<F: Debug> {
    terms: Vec<SimplexTerm<F>>,
    free: F,
    goal: Goal,
}

#[derive(Debug)]
pub struct SimplexTask<F: Debug> {
    restrictions: Vec<SimplexRestriction<F>>,
    target_fn: SimplexTarget<F>,
}

struct SimplexTaskParts<F: Debug> {
    a: Array2<F>,
    b: Array1<F>,
    z: Array1<F>,
}

pub struct Simple;
pub struct Taxes;
pub struct DoublePhase;

#[derive(Debug)]
pub struct CanonicSimplexTask<T: Debug, M> {
    task: SimplexTask<T>,
    max_index: u64,
    phantom: PhantomData<M>,
}

impl<T: Debug + From<Rational64>> From<Task> for SimplexTask<T> {
    fn from(value: Task) -> Self {
        let restrictions = value
            .restrictions
            .into_iter()
            .map(|x| SimplexRestriction {
                free: x.value.into(),
                relation: x.relation,
                terms: x
                    .terms
                    .into_iter()
                    .map(|y| SimplexTerm {
                        coef: y.coef.into(),
                        index: y.index,
                    })
                    .collect(),
            })
            .collect();

        let target_fn = SimplexTarget {
            free: value.target_fn.value.into(),
            terms: value
                .target_fn
                .terms
                .into_iter()
                .map(|x| SimplexTerm {
                    coef: x.coef.into(),
                    index: x.index,
                })
                .collect(),
            goal: value.target_fn.goal,
        };

        Self {
            restrictions,
            target_fn,
        }
    }
}

impl<T: Debug> SimplexTask<T> {
    pub fn canonize<M>(mut self) -> CanonicSimplexTask<T, M>
    where
        T: Num + NumAssign + PartialOrd,
    {
        let mut max_index = self
            .restrictions
            .iter()
            .flat_map(|x| &x.terms)
            .max_by_key(|x| x.index)
            .unwrap()
            .index;

        for restriction in &mut self.restrictions {
            match restriction.relation {
                Relation::Less => {
                    restriction.terms.push(SimplexTerm {
                        coef: T::one(),
                        index: max_index + 1,
                    });
                    max_index += 1;
                }
                Relation::Equal => (),
                Relation::Greater => {
                    restriction.terms.push(SimplexTerm {
                        coef: T::zero() - T::one(),
                        index: max_index + 1,
                    });
                    max_index += 1;
                }
            }

            restriction.relation = Relation::Equal;

            if restriction.free < T::zero() {
                restriction
                    .terms
                    .iter_mut()
                    .for_each(|x| x.coef *= T::zero() - T::one());
                restriction.free *= T::zero() - T::one();
            }
        }
        CanonicSimplexTask {
            task: self,
            max_index,
            phantom: PhantomData
        }
    }
}

#[cfg(not(feature = "taxes"))]
impl<F: Display + Num + Clone + Debug + Copy> From<CanonicSimplexTask<F, Simple>>
    for SimplexSolver<F>
{
    fn from(val: CanonicSimplexTask<F, Simple>) -> Self {
        let goal = val.task.target_fn.goal.clone();

        let mut parts = val.into_a_b_z();
        parts.invert_z();
        let contents = parts.into_contents();

        SimplexSolver::from_contents(contents, goal)
    }
}

impl<F: Display + Num + Clone + Debug + Copy> From<CanonicSimplexTask<Tax<F>, Taxes>>
    for SimplexSolver<Tax<F>>
{
    fn from(val: CanonicSimplexTask<Tax<F>, Taxes>) -> Self {
        let goal = val.task.target_fn.goal.clone();
        let mut parts = val.into_a_b_z();
        parts.add_taxes();
        parts.add_basis();
        parts.invert_z();
        let contents = parts.into_contents();

        SimplexSolver::from_contents(contents, goal)
    }
}

impl<F: Display + Num + Clone + Debug + Copy> From<CanonicSimplexTask<F, DoublePhase>>
    for SimplexSolver<F>
{
    fn from(val: CanonicSimplexTask<F, DoublePhase>) -> Self {
        let goal = val.task.target_fn.goal.clone();
        let mut parts = val.into_a_b_z();
        parts.add_basis();
        parts.invert_z();
        let contents = parts.into_contents();

        SimplexSolver::from_contents(contents, goal)
    }
}

impl<T: Debug, M> CanonicSimplexTask<T, M> {
    fn into_a_b_z(self) -> SimplexTaskParts<T>
    where
        T: Copy + Zero,
    {
        let restrictions_len = self.task.restrictions.len();

        let mut a_hash_map = self
            .task
            .restrictions
            .iter()
            .map(|x| {
                x.terms
                    .iter()
                    .map(|y| ((y.index - 1) as usize, y.coef))
                    .collect::<HashMap<_, _>>()
            })
            .enumerate()
            .collect::<HashMap<_, _>>();

        let mut z_hash_map = self
            .task
            .target_fn
            .terms
            .into_iter()
            .map(|x| ((x.index - 1) as usize, x.coef))
            .collect::<HashMap<_, _>>();

        let a = Array2::from_shape_fn((restrictions_len, self.max_index as usize), |(i, j)| {
            *a_hash_map
                .entry(i)
                .or_insert(HashMap::new())
                .entry(j)
                .or_insert(T::zero())
        });
        let b = Array1::from_shape_vec(
            restrictions_len,
            self.task.restrictions.into_iter().map(|x| x.free).collect(),
        )
        .unwrap();
        let mut z = Array1::from_shape_fn(self.max_index as usize, |i| {
            *z_hash_map.entry(i).or_insert(T::zero())
        });
        z.push(Axis(0), aview0(&self.task.target_fn.free)).unwrap();

        SimplexTaskParts { a, b, z }
    }
}

impl<T: Debug + Display + Num + Clone> SimplexTaskParts<Tax<T>> {
    fn add_taxes(&mut self)
    where
        T: Num + Clone + Display,
    {
        let mut taxed = self.a.sum_axis(Axis(0)).mapv(|x| x.into_tax());
        taxed
            .push(Axis(0), aview0(&self.b.sum().into_tax()))
            .unwrap();

        self.z
            .zip_mut_with(&taxed, |x, y| *x = x.clone() + y.clone())
    }
}

impl<T: Debug> SimplexTaskParts<T> {
    fn add_basis(&mut self)
    where
        T: Clone + Num,
    {
        let max_index = self.z.len() - 1;
        let restrictions_len = self.a.len_of(Axis(0));
        self.a
            .append(Axis(1), Array2::eye(restrictions_len).view())
            .unwrap();
        self.z
            .append(
                Axis(0),
                Array1::from_elem(restrictions_len, T::zero()).view(),
            )
            .unwrap();
        self.z.swap(max_index, max_index + restrictions_len);
    }

    fn invert_z(&mut self)
    where
        T: Num + Clone,
    {
        self.z
            .map_inplace(|x| *x = x.clone() * (T::zero() - T::one()));
    }

    fn into_contents(mut self) -> Array2<T>
    where
        T: Clone,
    {
        self.a.push_column(self.b.view()).unwrap();
        self.a.push_row(self.z.view()).unwrap();

        self.a
    }
}
