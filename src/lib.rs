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

pub use stencil::*;

use ndarray::*;

pub trait NdArray {
    type Elem: LinalgScalar;
    type Dim: Dimension;
    fn shape(&self) -> <Self::Dim as Dimension>::Pattern;
    fn as_view(&self) -> ArrayView<Self::Elem, Self::Dim>;
    fn as_view_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Dim>;
}

pub trait Creatable: Clone + NdArray {
    fn zeros(<Self::Dim as Dimension>::Pattern) -> Self;
}

/// Uniformly coordinated array
pub trait Manifold: NdArray {
    /// Type of coordinate
    type Coordinate;

    /// Increment of coordinate
    fn dx(&self) -> Self::Coordinate;

    /// Fill manifold by a function
    fn coordinate_fill<F>(&mut self, F)
    where
        F: Fn(Self::Coordinate) -> Self::Elem;

    /// Map values on manifold using a function
    fn coordinate_map<F>(&mut self, F)
    where
        F: Fn(Self::Coordinate, Self::Elem) -> Self::Elem;
}
