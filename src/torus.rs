//! Define N-dimensional torus

use super::*;
use num_traits::Float;

use std::f64::consts::PI;

/// N-dimensional torus
///
/// For simulations with the periodic boundary condition
#[derive(Clone)]
pub struct Torus<A: LinalgScalar, D: Dimension> {
    data: Array<A, D>,
}

impl<A, D> NdArray for Torus<A, D>
where
    A: LinalgScalar,
    D: Dimension,
{
    type Elem = A;
    type Dim = D;
    fn shape(&self) -> D::Pattern {
        self.data.dim()
    }
}

impl<A, D> Creatable for Torus<A, D>
where
    A: LinalgScalar,
    D: Dimension,
{
    fn zeros(p: D::Pattern) -> Self {
        Self {
            data: Array::zeros(p),
        }
    }
}

impl<A, D> Viewable for Torus<A, D>
where
    A: LinalgScalar,
    D: Dimension,
{
    fn as_view(&self) -> ArrayView<Self::Elem, Self::Dim> {
        self.data.view()
    }
    fn as_view_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Dim> {
        self.data.view_mut()
    }
}

impl<A: LinalgScalar> StencilArray<N1D1<A>> for Torus<A, Ix1> {
    fn stencil_map<Func>(&self, out: &mut Self, f: Func)
    where
        Func: Fn(N1D1<A>) -> Self::Elem,
    {
        let n = self.shape();
        for i in 0..n {
            let nn = N1D1 {
                l: self.data[(n + i - 1) % n],
                r: self.data[(i + 1) % n],
                c: self.data[i],
            };
            out.data[i] = f(nn);
        }
    }
}

impl<A: LinalgScalar> StencilArray<N2D1<A>> for Torus<A, Ix1> {
    fn stencil_map<Func>(&self, out: &mut Self, f: Func)
    where
        Func: Fn(N2D1<A>) -> Self::Elem,
    {
        let n = self.shape();
        for i in 0..n {
            let nn = N2D1 {
                ll: self.data[(n + i - 2) % n],
                rr: self.data[(i + 2) % n],
                l: self.data[(n + i - 1) % n],
                r: self.data[(i + 1) % n],
                c: self.data[i],
            };
            out.data[i] = f(nn);
        }
    }
}

impl<A: LinalgScalar> StencilArray<N1D2<A>> for Torus<A, Ix2> {
    fn stencil_map<Func>(&self, out: &mut Self, f: Func)
    where
        Func: Fn(N1D2<A>) -> Self::Elem,
    {
        let (n, m) = self.shape();
        for i in 0..n {
            for j in 0..m {
                let nn = N1D2 {
                    t: self.data[(i, (j + 1) % m)],
                    b: self.data[(i, (m + j - 1) % m)],
                    l: self.data[((n + i - 1) % n, j)],
                    r: self.data[((i + 1) % n, j)],
                    c: self.data[(i, j)],
                };
                out.data[(i, j)] = f(nn);
            }
        }
    }
}

impl<A: LinalgScalar + Float> Manifold for Torus<A, Ix1> {
    type Coordinate = A;

    fn dx(&self) -> Self::Coordinate {
        A::from(2.0 * PI / self.data.len() as f64).unwrap()
    }

    fn coordinate_fill<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate) -> Self::Elem,
    {
        let dx = self.dx();
        for (i, v) in self.data.iter_mut().enumerate() {
            let i = A::from(i).unwrap();
            *v = f(i * dx);
        }
    }

    fn coordinate_map<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate, Self::Elem) -> Self::Elem,
    {
        let dx = self.dx();
        for (i, v) in self.data.iter_mut().enumerate() {
            let i = A::from(i).unwrap();
            *v = f(i * dx, *v);
        }
    }
}

impl<A: LinalgScalar + Float> Manifold for Torus<A, Ix2> {
    type Coordinate = (A, A);

    fn dx(&self) -> Self::Coordinate {
        let (n, m) = self.shape();
        (
            A::from(2.0 * PI / n as f64).unwrap(),
            A::from(2.0 * PI / m as f64).unwrap(),
        )
    }

    fn coordinate_fill<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate) -> Self::Elem,
    {
        let (n, m) = self.shape();
        let (dx, dy) = self.dx();
        for i in 0..n {
            for j in 0..m {
                let x = A::from(i).unwrap() * dx;
                let y = A::from(j).unwrap() * dy;
                self.data[(i, j)] = f((x, y));
            }
        }
    }

    fn coordinate_map<F>(&mut self, f: F)
    where
        F: Fn(Self::Coordinate, Self::Elem) -> Self::Elem,
    {
        let (n, m) = self.shape();
        let (dx, dy) = self.dx();
        for i in 0..n {
            for j in 0..m {
                let x = A::from(i).unwrap() * dx;
                let y = A::from(j).unwrap() * dy;
                self.data[(i, j)] = f((x, y), self.data[(i, j)]);
            }
        }
    }
}
