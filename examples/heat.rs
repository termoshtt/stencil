extern crate asink;
extern crate ndarray;
extern crate stencil;

use ndarray::*;
use std::sync::mpsc::Sender;
use std::f64::consts::PI;
use asink::Sink;

use stencil::*;
use stencil::region::*;
use stencil::padding::P1;

fn periodic(r: Sender<Vec<f64>>) {
    let dt = 1e-3;
    let mut t = torus::Torus::<f64, Ix1>::zeros(32);
    t.coordinate_fill(|x| x.sin());
    let dx = t.dx();
    let mut s = t.clone();
    for _step in 0..3000 {
        t.stencil_map(&mut s, |n: N1D1<f64>| {
            let d2 = (n.l + n.r - 2.0 * n.c) / (dx.powi(2));
            n.c + dt * d2
        });
        r.send(s.as_view().to_vec()).unwrap();
        ::std::mem::swap(&mut t, &mut s);
    }
}

fn fixed(r: Sender<Vec<f64>>) {
    let dt = 1e-3;
    let mut t = Line::<f64, P1, Closed>::new(64, 0.0, 1.0, 2.5 * PI);
    t.coordinate_fill(|x| x.sin());
    let dx = t.dx();
    let mut s = t.clone();
    for _step in 0..10000 {
        t.stencil_map(&mut s, |n: N1D1<f64>| {
            let d2 = (n.l + n.r - 2.0 * n.c) / (dx.powi(2));
            n.c + dt * d2
        });
        r.send(s.as_view().to_vec()).unwrap();
        ::std::mem::swap(&mut t, &mut s);
    }
}

fn main() {
    let sink = asink::json::JsonSink::from_str("heat_periodic.json");
    let (r, th) = sink.run();
    periodic(r);
    th.join().unwrap();

    let sink = asink::json::JsonSink::from_str("heat_fixed.json");
    let (r, th) = sink.run();
    fixed(r);
    th.join().unwrap();
}
