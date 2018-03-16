extern crate asink;
extern crate ndarray;
extern crate stencil;

use stencil::*;
use ndarray::*;
use std::sync::mpsc::Sender;
use asink::Sink;

fn periodic(r: Sender<Vec<f64>>) {
    let dt = 1e-3;
    let mut t = torus::Torus::<f64, Ix1>::zeros(32);
    t.coordinate_fill(|x| x.sin());
    let dx = t.dx();
    let mut s = t.clone();
    for _step in 0..100_000 {
        t.stencil_map(&mut s, |n: N1D1<f64>| {
            let d2 = (n.l + n.r - 2.0 * n.c) / (dx.powi(2));
            n.c + dt * d2
        });
        r.send(s.as_view().to_vec()).unwrap();
        ::std::mem::swap(&mut t, &mut s);
    }
}

fn main() {
    let sink = asink::json::JsonSink::from_str("heat.json");
    let (r, th) = sink.run();
    periodic(r);
    th.join().unwrap();
}
