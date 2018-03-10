#[macro_use]
extern crate ndarray;

pub mod torus;

use ndarray::*;

pub trait NdArray {
    type Elem: LinalgScalar;
    type Dim: Dimension;
    fn shape(&self) -> <Self::Dim as Dimension>::Pattern;
}

pub trait Viewable: NdArray {
    fn as_view(&self) -> ArrayView<Self::Elem, Self::Dim>;
    fn as_view_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Dim>;
}

pub trait StencilArray<St>: NdArray
where
    St: Stencil<Elem = Self::Elem, Dim = Self::Dim>,
{
    /// Execute a stencil calculation
    fn stencil_map<Func>(&self, out: &mut Self, Func)
    where
        Func: Fn(St) -> Self::Elem;
}

pub trait Stencil {
    type Elem: LinalgScalar;
    type Dim: Dimension;
}

#[derive(Clone, Copy)]
pub struct N1D2<A: Clone + Copy> {
    pub t: A, // top
    pub b: A, // bottom
    pub l: A, // left
    pub r: A, // right
    pub c: A, // center
}

impl<A: LinalgScalar + Clone + Copy> Stencil for N1D2<A> {
    type Elem = A;
    type Dim = Ix2;
}
