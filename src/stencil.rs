use super::*;

use ndarray::*;
use padding::*;

/// Array with stencil calculations
pub trait StencilArray<St>: NdArray
where
    St: Stencil<Elem = Self::Elem, Dim = Self::Dim>,
{
    /// Execute a stencil calculation
    fn stencil_map<Output, Func>(&self, out: &mut Output, Func)
    where
        Output: NdArray<Dim = Self::Dim>,
        Func: Fn(St) -> Output::Elem;
}

pub trait Stencil {
    type Elem: LinalgScalar;
    type Dim: Dimension;
    type Padding: Padding;
}

/// one-neighbor, one-dimensional stencil
#[derive(Clone, Copy)]
pub struct N1D1<A: LinalgScalar> {
    /// left
    pub l: A,
    /// right
    pub r: A,
    /// center
    pub c: A,
}

impl<A: LinalgScalar> Stencil for N1D1<A> {
    type Elem = A;
    type Dim = Ix1;
    type Padding = P1;
}

/// two-neighbor, one-dimensional stencil
#[derive(Clone, Copy)]
pub struct N2D1<A: LinalgScalar> {
    /// left
    pub l: A,
    /// right
    pub r: A,
    /// left of left
    pub ll: A,
    /// right of right
    pub rr: A,
    /// center
    pub c: A,
}

impl<A: LinalgScalar> Stencil for N2D1<A> {
    type Elem = A;
    type Dim = Ix1;
    type Padding = P2;
}

/// one-neighbor, two-dimensional stencil
#[derive(Clone, Copy)]
pub struct N1D2<A: LinalgScalar> {
    /// top
    pub t: A,
    /// bottom
    pub b: A,
    /// left
    pub l: A,
    /// right
    pub r: A,
    /// center
    pub c: A,
}

impl<A: LinalgScalar> Stencil for N1D2<A> {
    type Elem = A;
    type Dim = Ix2;
    type Padding = P1;
}
