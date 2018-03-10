use super::*;

/// Two-dimensional torus
#[derive(Clone)]
pub struct Torus2<A: LinalgScalar + Clone + Copy> {
    data: Array2<A>,
}

impl<A: LinalgScalar + Clone + Copy> Torus2<A> {
    pub fn zeros(n: usize, m: usize) -> Self {
        Self {
            data: Array::zeros((n + 2, m + 2)),
        }
    }

    fn fill_periodic(&mut self) {
        let (n, m) = self.shape();
        for j in 0..m {
            self.data[(0, j + 1)] = self.data[(n - 2, j + 1)];
            self.data[(n - 1, j + 1)] = self.data[(1, j + 1)];
        }
        for i in 0..n {
            self.data[(i + 1, 0)] = self.data[(i + 1, m - 2)];
            self.data[(i + 1, m - 1)] = self.data[(i + 1, 1)];
        }
    }

    fn st_map_core<B, F>(&self, out: &mut Torus2<B>, func: F)
    where
        B: LinalgScalar,
        F: Fn(N1D2<A>) -> B,
    {
        let (n, m) = self.shape();
        let data = self.data.as_slice().unwrap();
        let out = out.data.as_slice_mut().unwrap();
        for i in 0..n {
            for j in 0..m {
                let neighbor = N1D2 {
                    t: data[(i + 0) * (m + 2) + (j + 0)],
                    b: data[(i + 2) * (m + 2) + (j + 0)],
                    l: data[(i + 1) * (m + 2) + (j + 0)],
                    r: data[(i + 1) * (m + 2) + (j + 2)],
                    c: data[(i + 1) * (m + 2) + (j + 1)],
                };
                out[(i + 1) * (m + 2) + (j + 1)] = func(neighbor);
            }
        }
    }
}

impl<A: LinalgScalar> NdArray for Torus2<A> {
    type Elem = A;
    type Dim = Ix2;
    fn shape(&self) -> (usize, usize) {
        let (n, m) = self.data.dim();
        (n - 2, m - 2)
    }
}

impl<A: LinalgScalar> StencilArray<N1D2<A>> for Torus2<A> {
    fn stencil_map<F>(&self, out: &mut Self, func: F)
    where
        F: Fn(N1D2<A>) -> A,
    {
        self.st_map_core(out, func);
        out.fill_periodic();
    }
}

impl<A: LinalgScalar> Viewable for Torus2<A> {
    fn as_view(&self) -> ArrayView2<A> {
        self.data.slice(s![1..-1, 1..-1])
    }

    fn as_view_mut(&mut self) -> ArrayViewMut2<A> {
        self.data.slice_mut(s![1..-1, 1..-1])
    }
}
