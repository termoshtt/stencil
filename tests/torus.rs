extern crate ndarray;
extern crate ndarray_linalg;
extern crate stencil;

use stencil::*;
use ndarray::*;
use ndarray_linalg::*;

// Test central difference of `sin(kx)` to be `-k cos(kx)`
#[test]
fn central_diff() {
    let n = 128;
    let mut a = torus::Torus::<f64, Ix1>::zeros(n);
    let mut b = a.clone();
    a.coordinate_fill(|x| x.sin());
    let dx = a.dx();
    a.stencil_map(&mut b, |n: N1D1<f64>| (n.r - n.l) / (2.0 * dx));
    a.coordinate_fill(|x| x.cos());
    close_l2(&a.as_view(), &b.as_view(), 1e-3).unwrap();
}
