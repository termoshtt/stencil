use super::*;
use num_traits::Float;

pub fn cfill_1d<A, V, F>(a: &mut V, dx: A, f: F)
where
    A: LinalgScalar + Float,
    V: Viewable<Elem = A, Dim = Ix1>,
    F: Fn(A) -> A,
{
    for (i, v) in a.as_view_mut().iter_mut().enumerate() {
        let i = A::from(i).unwrap();
        *v = f(i * dx);
    }
}

pub fn cmap_1d<A, V, F>(a: &mut V, dx: A, f: F)
where
    A: LinalgScalar + Float,
    V: Viewable<Elem = A, Dim = Ix1>,
    F: Fn(A, A) -> A,
{
    for (i, v) in a.as_view_mut().iter_mut().enumerate() {
        let i = A::from(i).unwrap();
        *v = f(i * dx, *v);
    }
}