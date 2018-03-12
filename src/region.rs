use super::*;
use num_traits::Float;

pub struct Line<A: LinalgScalar> {
    data: Array1<A>,
    padding: usize,
    left: A,
    right: A,
    length: A,
}

impl<A: LinalgScalar> Line<A> {
    pub fn new(n: usize, padding: usize, left: A, right: A, length: A) -> Self {
        Self {
            data: Array::zeros(n + 2 * padding),
            padding,
            left,
            right,
            length,
        }
    }

    fn fill_boundary(&mut self) {
        let n = self.data.len();
        for i in 0..self.padding {
            self.data[i] = self.left;
            self.data[n - i] = self.right;
        }
    }
}

impl<A: LinalgScalar> NdArray for Line<A> {
    type Elem = A;
    type Dim = Ix1;

    fn shape(&self) -> usize {
        self.data.len() - 2 * self.padding
    }
}

impl<A: LinalgScalar> Viewable for Line<A> {
    fn as_view(&self) -> ArrayView1<A> {
        let p = self.padding as i32;
        self.data.slice(s![p..-p])
    }
    fn as_view_mut(&mut self) -> ArrayViewMut1<A> {
        let p = self.padding as i32;
        self.data.slice_mut(s![p..-p])
    }
}

impl<A: LinalgScalar + Float> Manifold for Line<A> {
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
