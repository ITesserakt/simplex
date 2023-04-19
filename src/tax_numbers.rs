use std::{
    fmt::{Display, Debug},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
    str::FromStr,
};

use num::{traits::NumAssign, Complex, Num, One, Zero};

#[derive(PartialEq, Clone, Copy, Eq)]
pub struct Tax<T>(Complex<T>); // T + T * M

impl<T> Tax<T> {
    pub fn into_tax(self) -> Tax<T> where T: Zero {
        Tax(Complex { re: T::zero(), im: self.0.re })
    }
}

impl<T: PartialOrd> PartialOrd for Tax<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.0
                .im
                .partial_cmp(&other.0.im)?
                .then(self.0.re.partial_cmp(&other.0.re)?),
        )
    }
}

impl<T: Ord> Ord for Tax<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.im.cmp(&other.0.im).then(self.0.re.cmp(&other.0.re))
    }
}

impl<T> From<(T, T)> for Tax<T> {
    fn from(value: (T, T)) -> Self {
        Self(Complex::new(value.0, value.1))
    }
}

impl<T: Zero> From<T> for Tax<T> {
    fn from(value: T) -> Self {
        Self(Complex {
            re: value,
            im: T::zero(),
        })
    }
}

impl<T: FromStr + One + Zero> FromStr for Tax<T> {
    type Err = T::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "M" => Ok((T::zero(), T::one()).into()),
            _ => Ok((s.parse()?, T::zero()).into()),
        }
    }
}

macro_rules! functor_like_impl {
    ($t:ident, $op:ident) => {
        impl<T: Num + Clone> $t<Tax<T>> for Tax<T> {
            type Output = Tax<T>;

            fn $op(self, rhs: Tax<T>) -> Self::Output {
                Tax($t::$op(self.0, rhs.0))
            }
        }
    };
}

macro_rules! functor_like_self_impl {
    ($t:ident, $op:ident) => {
        impl<T: NumAssign + Clone> $t<Tax<T>> for Tax<T> {
            fn $op(&mut self, rhs: Tax<T>) {
                $t::$op(&mut self.0, rhs.0)
            }
        }
    };
}

functor_like_impl!(Add, add);
functor_like_impl!(Mul, mul);
functor_like_impl!(Sub, sub);
functor_like_impl!(Div, div);
functor_like_impl!(Rem, rem);
functor_like_self_impl!(AddAssign, add_assign);
functor_like_self_impl!(MulAssign, mul_assign);
functor_like_self_impl!(SubAssign, sub_assign);
functor_like_self_impl!(DivAssign, div_assign);
functor_like_self_impl!(RemAssign, rem_assign);

impl<T: num::Num + std::clone::Clone> One for Tax<T> {
    fn one() -> Self {
        (T::one(), T::zero()).into()
    }
}

impl<T: std::clone::Clone + num::Num> Zero for Tax<T> {
    fn zero() -> Self {
        (T::zero(), T::zero()).into()
    }

    fn is_zero(&self) -> bool {
        self.0.re.is_zero() && self.0.im.is_zero()
    }
}

impl<T: Num + std::clone::Clone> Num for Tax<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(match str {
            "M" => (T::zero(), T::one()).into(),
            _ => (T::from_str_radix(str, radix)?, T::zero()).into(),
        })
    }
}

impl<T: Display + Num + Clone> Display for Tax<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else if self.0.re.is_zero() {
            write!(f, "{}M", self.0.im)
        } else if self.0.im.is_zero() {
            write!(f, "{}", self.0.re)
        } else {
            write!(f, "{} + {}M", self.0.re, self.0.im)
        }
    }
}

impl<T: Display + Num + Clone> Debug for Tax<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else if self.0.re.is_zero() {
            write!(f, "{}M", self.0.im)
        } else if self.0.im.is_zero() {
            write!(f, "{}", self.0.re)
        } else {
            write!(f, "{} + {}M", self.0.re, self.0.im)
        }
    }
}
