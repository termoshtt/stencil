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
        let nn = N1D1 {
            l: self.data[n - 1],
            r: self.data[1],
            c: self.data[0],
        };
        out.data[0] = f(nn);
        for i in 1..(n - 1) {
            let nn = N1D1 {
                l: self.data[i - 1],
                r: self.data[i + 1],
                c: self.data[i],
            };
            out.data[i] = f(nn);
        }
        let nn = N1D1 {
            l: self.data[n - 2],
            r: self.data[0],
            c: self.data[n - 1],
        };
        out.data[n - 1] = f(nn);
    }
}
