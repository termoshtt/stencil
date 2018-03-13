extern crate ndarray;
extern crate ndarray_linalg;
extern crate stencil;

use ndarray_linalg::*;
use std::f64::consts::PI;

use stencil::*;
use stencil::region::*;
use stencil::padding::*;

#[test]
fn closed() {
    let n = 128;
    let mut a = region::Line::<f64, P1, Closed>::new(n, 0.0, 0.0, 2.0 * PI);
    a.coordinate_fill(|x| x.sin());
    let dx = a.dx();
    let mut b = a.clone();
    a.stencil_map(&mut b, |n: N1D1<f64>| (n.r - n.l) / (2.0 * dx));
    b.fill_edge();
    a.coordinate_fill(|x| x.cos());
    close_l2(&a.as_view(), &b.as_view(), 1e-1).unwrap();
}

#[test]
fn open() {
    let n = 128;
    let mut a = region::Line::<f64, P1, Open>::new(n, 0.0, 0.0, 2.0 * PI);
    a.coordinate_fill(|x| x.sin());
    let dx = a.dx();
    let mut b = a.clone();
    a.stencil_map(&mut b, |n: N1D1<f64>| (n.r - n.l) / (2.0 * dx));
    a.coordinate_fill(|x| x.cos());
    close_l2(&a.as_view(), &b.as_view(), 1e-3).unwrap();
}
