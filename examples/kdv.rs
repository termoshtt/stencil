extern crate asink;
#[macro_use]
extern crate ndarray;
extern crate stencil;

use stencil::*;
use ndarray::*;
use asink::Sink;

use std::sync::mpsc::Sender;
use std::mem::swap;

fn kdv(r: Sender<Vec<f64>>) {
    let dt = 1e-5;
    let mut t = torus::Torus::<f64, Ix1>::zeros(128);
    t.coordinate_fill(|x| x.sin());
    let dx = t.dx();
    let mut b = t.clone();
    let mut n = t.clone();
    for step in 0..1_000_000 {
        t.stencil_map(&mut n, |n: N2D1<f64>| {
            // ZK scheme
            let uux = (n.l + n.c + n.r) * (n.r - n.l) / (3.0 * dx);
            let u3 = (n.rr - 2.0 * n.r + 2.0 * n.l - n.ll) / dx.powi(3);
            uux + 0.01 * u3
        });
        azip!(mut n, b in { *n = b - dt * *n });
        if step % 1000 == 0 {
            r.send(n.as_view().to_vec()).unwrap();
        }
        swap(&mut t, &mut b);
        swap(&mut t, &mut n);
    }
}

fn main() {
    let sink = asink::json::JsonSink::from_str("kdv.json");
    let (r, th) = sink.run();
    kdv(r);
    th.join().unwrap();
}
