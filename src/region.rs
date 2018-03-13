//! Line regions for fixed boundary condition
//!
//! Data Layout
//! ------------
//!
//! - Open: `x_i = (i+1)L / (N+1)`
//! - Closed: `x_i = iL / (N-1)`, with fixed values `x_0 = x_L`, `x_{N-1} = x_R`
//!
//! These are used for implementing `Manifold` trait.
//!
//! ```ignore
//! xxx---------xxx  Line<A, P3, Open>(N=9)
//! xxo---------oxx  Line<A, P2, Closed>(N=11)
//!   |<-- L -->|    dx = L/10
//! ```
//!
//! Examples
//! --------
//!
//! ```
//! # extern crate ndarray;
//! # extern crate stencil;
//! # use stencil::*;
//! # use stencil::padding::*;
//! # use stencil::region::*;
//! # fn main() {
//! // closed line with one-padding:
//! let mut a = Line::<f64, P1, Closed>::new(128, 0.0, 0.0, 2.0);
//! // open line with two-padding:
//! let mut a = Line::<f64, P2, Open>::new(128, 0.0, 0.0, 2.0);
//! # }
//! ```
use super::*;
use super::padding::*;

use ndarray::*;
use num_traits::Float;
use std::marker::PhantomData;

pub trait Edge: Clone + Copy {
    fn len() -> usize;
}
#[derive(Clone, Debug, Copy)]
pub struct Open {}
#[derive(Clone, Debug, Copy)]
pub struct Closed {}
impl Edge for Open {
    fn len() -> usize {
        1
    }
}
impl Edge for Closed {
    fn len() -> usize {
        0
    }
}
///
#[derive(Debug, Clone)]
pub struct Line<A: LinalgScalar, P: Padding, E: Edge> {
    data: Array1<A>,
    left: A,
    right: A,
    length: A,
    phantom: PhantomData<(P, E)>,
}

impl<A: LinalgScalar, P: Padding, E: Edge> Line<A, P, E> {
    pub fn new(n: usize, left: A, right: A, length: A) -> Self {
        Self {
            data: Array::zeros(n + 2 * P::len()),
            left,
            right,
            length,
            phantom: PhantomData,
        }
    }

    pub fn fill_edge(&mut self) {
        let n = self.data.len();
        for i in 0..P::len() + E::len() - 1 {
            self.data[i] = self.left;
            self.data[n - 1 - i] = self.right;
        }
    }
}

impl<A: LinalgScalar, P: Padding, E: Edge> NdArray for Line<A, P, E> {
    type Elem = A;
    type Dim = Ix1;

    fn shape(&self) -> usize {
        self.data.len() - 2 * P::len()
    }

    fn as_view(&self) -> ArrayView1<A> {
        let p = P::len() as i32;
        self.data.slice(s![p..-p])
    }

    fn as_view_mut(&mut self) -> ArrayViewMut1<A> {
        let p = P::len() as i32;
        self.data.slice_mut(s![p..-p])
    }
}

impl<A: LinalgScalar + Float, P: Padding, E: Edge> Manifold for Line<A, P, E> {
    type Coordinate = A;
    fn dx(&self) -> A {
        self.length / A::from(self.shape() + 2 * E::len() - 1).unwrap()
    }

    fn coordinate_fill<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate) -> Self::Elem,
    {
        let dx = self.dx();
        for (i, v) in self.as_view_mut().iter_mut().enumerate() {
            let x = dx * A::from(i + E::len()).unwrap();
            *v = f(x);
        }
        self.fill_edge();
    }

    fn coordinate_map<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate, Self::Elem) -> Self::Elem,
    {
        let dx = self.dx();
        for (i, v) in self.as_view_mut().iter_mut().enumerate() {
            let i = A::from(i + 1).unwrap();
            *v = f(i * dx, *v);
        }
        self.fill_edge();
    }
}

impl<A, P, E> StencilArray<N1D1<A>> for Line<A, P, E>
where
    A: LinalgScalar,
    E: Edge,
    P: GreaterEq<<N1D1<A> as Stencil>::Padding>,
{
    fn stencil_map<Output, Func>(&self, out: &mut Output, f: Func)
    where
        Output: NdArray<Dim = Self::Dim>,
        Func: Fn(N1D1<A>) -> Output::Elem,
    {
        let n = self.shape();
        let mut out = out.as_view_mut();
        for i in 0..n {
            let j = i + P::len();
            let nn = N1D1 {
                l: self.data[(j - 1)],
                r: self.data[(j + 1)],
                c: self.data[j],
            };
            out[i] = f(nn);
        }
    }
}
