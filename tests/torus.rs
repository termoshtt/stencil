extern crate ndarray;
extern crate ndarray_linalg;
extern crate stencil;

use stencil::*;
use ndarray::*;
use ndarray_linalg::*;

use std::f64::consts::PI;

// Test central difference of `sin(kx)` to be `-k cos(kx)`
#[test]
fn central_diff() {
    let n = 128;
    let mut a = torus::Torus::<f64, Ix1>::zeros(n);
    let mut b = a.clone();
    let k0 = 2.0 * PI / n as f64;
    for (i, v) in a.as_view_mut().iter_mut().enumerate() {
        *v = (i as f64 * k0).sin();
    }
    a.stencil_map(&mut b, |n: N1D1<f64>| (n.r - n.l) * 0.5);
    for (i, v) in a.as_view_mut().iter_mut().enumerate() {
        *v = (i as f64 * k0).cos() * k0;
    }
    close_l2(&a.as_view(), &b.as_view(), 1e-3).unwrap();
}
