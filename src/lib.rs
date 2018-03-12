//! Stencil calculations
//!
//! One-dimensional case
//! --------------------
//!
//! Example to calculate the central-difference of `sin(x)` function
//! using [N1D1](struct.N1D1.html) stencil and [Torus](torus/struct.Torus.html)
//!
//! ```
//! # extern crate ndarray;
//! # extern crate stencil;
//! # use stencil::*;
//! # use ndarray::*;
//! # fn main() {
//! # let n = 128;
//! let mut a = torus::Torus::<f64, Ix1>::zeros(n);
//! let mut b = a.clone();
//! a.coordinate_fill(|x| x.sin());
//! let dx = a.dx();
//! a.stencil_map(&mut b, |n: N1D1<f64>| (n.r - n.l) / (2.0 * dx));
//! # }
//! ```
//!
//! Two-dimensional case
//! --------------------
//!
//! Example to calculate the central-difference of `sin(x)cos(y)` function
//! using [N1D2](struct.N1D2.html) stencil and [Torus](torus/struct.Torus.html)
//!
//! ```
//! # extern crate ndarray;
//! # extern crate stencil;
//! # use stencil::*;
//! # use ndarray::*;
//! # fn main() {
//! # let n = 128;
//! # let m = 64;
//! let mut a = torus::Torus::<f64, Ix2>::zeros((n, m));
//! a.coordinate_fill(|(x, y)| x.sin() * y.cos());
//! let (dx, dy) = a.dx();
//! let mut ax = a.clone();
//! let mut ay = a.clone();
//! a.stencil_map(&mut ax, |n: N1D2<f64>| (n.r - n.l) / (2.0 * dx));
//! a.stencil_map(&mut ay, |n: N1D2<f64>| (n.t - n.b) / (2.0 * dy));
//! # }
//! ```

#[macro_use]
extern crate ndarray;
extern crate num_traits;

pub mod padding;
pub mod region;
pub mod torus;
mod impl_util;
mod stencil;
mod array;

pub use stencil::*;
pub use array::*;
