pub trait Padding {
    fn len() -> usize;
}

pub trait LessEq<P: Padding>: Padding {}
pub trait GreaterEq<P: Padding>: Padding {}
pub trait EqualTo<P: Padding>: Padding {}

pub struct P1 {}
pub struct P2 {}
pub struct P3 {}

impl LessEq<P1> for P1 {}
impl LessEq<P2> for P1 {}
impl LessEq<P2> for P2 {}
impl LessEq<P3> for P1 {}
impl LessEq<P3> for P2 {}
impl LessEq<P3> for P3 {}

impl GreaterEq<P1> for P1 {}
impl GreaterEq<P1> for P2 {}
impl GreaterEq<P1> for P3 {}
impl GreaterEq<P2> for P2 {}
impl GreaterEq<P2> for P3 {}
impl GreaterEq<P3> for P3 {}

impl<Pa: LessEq<Pb> + GreaterEq<Pb>, Pb: Padding> EqualTo<Pb> for Pa {}

impl Padding for P1 {
    fn len() -> usize {
        1
    }
}

impl Padding for P2 {
    fn len() -> usize {
        2
    }
}

impl Padding for P3 {
    fn len() -> usize {
        3
    }
}
