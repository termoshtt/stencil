use ndarray::*;

pub trait NdArray {
    type Elem: LinalgScalar;
    type Dim: Dimension;
    fn shape(&self) -> <Self::Dim as Dimension>::Pattern;
    fn as_view(&self) -> ArrayView<Self::Elem, Self::Dim>;
    fn as_view_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Dim>;
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

impl<A: LinalgScalar, S: DataMut<Elem = A>, D: Dimension> NdArray for ArrayBase<S, D> {
    type Elem = A;
    type Dim = D;

    fn shape(&self) -> <Self::Dim as Dimension>::Pattern {
        self.dim()
    }

    fn as_view(&self) -> ArrayView<Self::Elem, Self::Dim> {
        self.view()
    }

    fn as_view_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Dim> {
        self.view_mut()
    }
}
