//! Typed Padding

/// Trait for typed paddings
pub trait Padding: Clone + Copy {
    /// Length of padding
    fn len() -> usize;
}

/// padding one element
#[derive(Clone, Debug, Copy)]
pub struct P1 {}

/// padding two element
#[derive(Clone, Debug, Copy)]
pub struct P2 {}

/// padding three element
#[derive(Clone, Debug, Copy)]
pub struct P3 {}

/// Compare padding sizes
pub trait LessEq<P: Padding>: Padding {}

impl LessEq<P1> for P1 {}
impl LessEq<P2> for P1 {}
impl LessEq<P2> for P2 {}
impl LessEq<P3> for P1 {}
impl LessEq<P3> for P2 {}
impl LessEq<P3> for P3 {}

/// Compare padding sizes
pub trait GreaterEq<P: Padding>: Padding {}

impl GreaterEq<P1> for P1 {}
impl GreaterEq<P1> for P2 {}
impl GreaterEq<P1> for P3 {}
impl GreaterEq<P2> for P2 {}
impl GreaterEq<P2> for P3 {}
impl GreaterEq<P3> for P3 {}

/// Compare padding sizes
pub trait EqualTo<P: Padding>: Padding {}

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
