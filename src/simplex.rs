extern crate ndarray;

use ndarray::{prelude::*, LinalgScalar};
use num::{traits::NumAssign, Num, Zero};
use std::{fmt::{Display}, ops::Div};

use crate::{errors::SimplexMethodError};

#[allow(dead_code)]
pub enum Aim {
    Minimize,
    Maximize,
}

pub struct SimplexSolver<N> {
    _contents: Array2<N>,
    basis: Array1<usize>,
    aim: Aim,
}

pub struct Solution<N> {
    basis_coeffs: Array1<(usize, N)>,
    coefficients: Array1<N>,
}

impl<F: Display + Num + NumAssign + Copy> Display for Solution<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let xs = self.coefficients.slice(s![..-1]);
        let free_z = self.coefficients.slice(s![-1]);

        let mut optimal_z = *free_z.into_scalar();
        for &(i, item) in &self.basis_coeffs {
            optimal_z += xs[i] * item;
        }

        writeln!(f, "Optimal z is: {}", optimal_z)?;
        writeln!(f, "Free variables are equal to: ")?;
        for &(i, item) in &self.basis_coeffs {
            writeln!(f, "   x{} = {item}", i + 1)?;
        }
        writeln!(f,)?;

        Ok(())
    }
}

impl<F> SimplexSolver<F> {
    #[inline]
    fn z(&self) -> ArrayView1<F> {
        self._contents.slice(s![-1, ..])
    }

    #[inline]
    fn a(&self) -> ArrayView2<F> {
        self._contents.slice(s![..-1, ..-1])
    }

    #[inline]
    fn b(&self) -> ArrayView1<F> {
        self._contents.slice(s![..-1, -1])
    }

    /// # Panics
    /// If either `N` or `M` is zero
    #[allow(dead_code)]
    pub fn from_canonical_matrix<const N: usize, const M: usize>(
        input: [[F; N]; M],
        z: [F; N],
        aim: Aim,
    ) -> Self
    where
        F: Clone + Zero,
    {
        if input.is_empty() {
            panic!("Given zero restrictions");
        }
        if z.is_empty() {
            panic!("No variables to solve for");
        }

        let mut matrix = Array2::from_shape_vec((M, N), input.flatten().to_vec()).unwrap();
        let z = Array1::from_vec(z.to_vec());

        matrix.push_row(z.view()).unwrap();

        Self {
            _contents: matrix,
            basis: z
                .slice(s![..-1])
                .indexed_iter()
                .filter(|x| x.1.is_zero())
                .map(|x| x.0)
                .collect(),
            aim,
        }
    }

    pub fn from_contents(contents: Array2<F>, aim: Aim) -> SimplexSolver<F>
    where
        F: Zero + Clone,
    {
        if contents.len_of(Axis(0)) == 0 {
            panic!("Given zero restrictions")
        }

        let z = contents.slice(s![-1, ..-1]).to_owned();

        Self {
            _contents: contents,
            basis: z
                .indexed_iter()
                .filter(|x| x.1.is_zero())
                .map(|x| x.0)
                .collect(),
            aim,
        }
    }

    fn is_optimal(&self) -> bool
    where
        F: Zero + PartialOrd,
    {
        match self.aim {
            Aim::Minimize => self.z().iter().all(|x| *x <= F::zero()),
            Aim::Maximize => self.z().iter().all(|x| *x >= F::zero()),
        }
    }

    fn pivot_column(&self) -> Result<usize, SimplexMethodError>
    where
        F: Zero + Ord + Copy,
    {
        let z = self.z();

        match self.aim {
            Aim::Minimize => z
                .indexed_iter()
                .take(self.z().len() - 1)
                .filter(|(_, x)| **x > F::zero())
                .max_by_key(|x| x.1),
            Aim::Maximize => z
                .indexed_iter()
                .take(self.z().len() - 1)
                .filter(|(_, x)| **x < F::zero())
                .min_by_key(|x| x.1),
        }
        .map(|x| x.0)
        .ok_or(SimplexMethodError::NoSolutions)
    }

    fn pivot_row(&self, pivot_col: usize) -> Result<usize, SimplexMethodError>
    where
        F: Zero + Ord + Div<F, Output = F> + Copy,
    {
        self.a()
            .column(pivot_col)
            .indexed_iter()
            .zip(self.b())
            .filter(|((_, x), _)| !x.is_zero())
            .map(|((i, x), y)| (i, *y / *x))
            .filter(|(_, x)| !x.is_zero() && *x > F::zero())
            .min_by_key(|x| x.1)
            .map(|x| x.0)
            .ok_or(SimplexMethodError::NoLimit)
    }

    fn pivot(&self) -> Result<(usize, usize, F), SimplexMethodError>
    where
        F: Zero + Ord + Div<F, Output = F> + Copy,
    {
        let col = self.pivot_column()?;
        let row = self.pivot_row(col)?;

        Ok((row, col, self._contents[(row, col)]))
    }
}

impl<T> SimplexSolver<T>
where
    T: Ord + Copy + LinalgScalar + Num + NumAssign + Display,
{
    fn make_iteration(&mut self) -> Result<(), SimplexMethodError> {
        let (p_row, p_col, pivot) = self.pivot()?;

        let mut pivot_row = self._contents.row_mut(p_row);
        pivot_row.map_inplace(|x| *x /= pivot);
        let pivot_row = self._contents.row(p_row).to_owned();

        for (i, mut row) in self._contents.rows_mut().into_iter().enumerate() {
            if i == p_row {
                continue;
            }

            let pivot_coeff = row[p_col];

            row.scaled_add(T::zero() - pivot_coeff, &pivot_row);
        }

        self.basis[p_row] = p_col;

        Ok(())
    }

    pub fn solve(mut self) -> Result<Solution<T>, SimplexMethodError> {
        while !self.is_optimal() {
            self.debug_state();
            self.make_iteration()?;
        }
        self.debug_state();

        let basis_coeffs = self
            .basis
            .iter()
            .zip(self.b())
            .map(|(i, x)| (*i, *x))
            .collect();
        let solution = self._contents.slice_move(s![-1, ..]);

        Ok(Solution {
            basis_coeffs,
            coefficients: solution,
        })
    }

    fn debug_state(&self) {
        for row in self._contents.outer_iter() {
            for item in &row {
                print!("{:<14} ", item.to_string());
            }
            println!();
        }
        println!("Basic: {}", self.basis);
    }
}
