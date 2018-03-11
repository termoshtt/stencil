extern crate ndarray;
extern crate ndarray_linalg;
extern crate stencil;

use stencil::*;
use ndarray::*;
use ndarray_linalg::*;

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

#[test]
fn diff2d() {
    let n = 128;
    let m = 64;
    let mut a = torus::Torus::<f64, Ix2>::zeros((n, m));
    a.coordinate_fill(|(x, y)| x.sin() * y.cos());

    let (dx, dy) = a.dx();
    let mut ax = a.clone();
    let mut ay = a.clone();
    a.stencil_map(&mut ax, |n: N1D2<f64>| (n.r - n.l) / (2.0 * dx));
    a.stencil_map(&mut ay, |n: N1D2<f64>| (n.t - n.b) / (2.0 * dy));

    let mut ax_ = a.clone();
    let mut ay_ = a.clone();
    ax_.coordinate_fill(|(x, y)| x.cos() * y.cos());
    ay_.coordinate_fill(|(x, y)| -x.sin() * y.sin());
    close_l2(&ax.as_view(), &ax_.as_view(), 1e-2).unwrap();
    close_l2(&ay.as_view(), &ay_.as_view(), 1e-2).unwrap();
}
