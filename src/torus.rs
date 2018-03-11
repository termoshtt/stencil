use super::*;

/// N-dimensional torus
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
