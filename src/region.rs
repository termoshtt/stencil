use super::{*, padding::*};
use num_traits::Float;
use std::marker::PhantomData;

pub struct Line<A: LinalgScalar, P: Padding> {
    data: Array1<A>,
    left: A,
    right: A,
    length: A,
    phantom: PhantomData<P>,
}

impl<A: LinalgScalar, P: Padding> Line<A, P> {
    pub fn new(n: usize, left: A, right: A, length: A) -> Self {
        Self {
            data: Array::zeros(n + 2 * P::len()),
            left,
            right,
            length,
            phantom: PhantomData,
        }
    }

    fn fill_boundary(&mut self) {
        let n = self.data.len();
        for i in 0..P::len() {
            self.data[i] = self.left;
            self.data[n - i] = self.right;
        }
    }
}

impl<A: LinalgScalar, P: Padding> NdArray for Line<A, P> {
    type Elem = A;
    type Dim = Ix1;

    fn shape(&self) -> usize {
        self.data.len() - 2 * P::len()
    }
}

impl<A: LinalgScalar, P: Padding> Viewable for Line<A, P> {
    fn as_view(&self) -> ArrayView1<A> {
        let p = P::len() as i32;
        self.data.slice(s![p..-p])
    }
    fn as_view_mut(&mut self) -> ArrayViewMut1<A> {
        let p = P::len() as i32;
        self.data.slice_mut(s![p..-p])
    }
}

impl<A: LinalgScalar + Float, P: Padding> Manifold for Line<A, P> {
    type Coordinate = A;
    fn dx(&self) -> A {
        self.length / A::from(self.shape()).unwrap()
    }

    fn coordinate_fill<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate) -> Self::Elem,
    {
        let dx = self.dx();
        impl_util::cfill_1d(self, dx, f);
        self.fill_boundary();
    }

    fn coordinate_map<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate, Self::Elem) -> Self::Elem,
    {
        let dx = self.dx();
        impl_util::cmap_1d(self, dx, f);
        self.fill_boundary();
    }
}

impl<A, P> StencilArray<N1D1<A>> for Line<A, P>
where
    A: LinalgScalar,
    P: GreaterEq<<N1D1<A> as Stencil>::Padding>,
{
    fn stencil_map<Func>(&self, out: &mut Self, f: Func)
    where
        Func: Fn(N1D1<A>) -> Self::Elem,
    {
        let n = self.shape();
        for i in 0..n {
            let i = i + P::len();
            let nn = N1D1 {
                l: self.data[(i - 1)],
                r: self.data[(i + 1)],
                c: self.data[i],
            };
            out.data[i] = f(nn);
        }
        out.fill_boundary();
    }
}
