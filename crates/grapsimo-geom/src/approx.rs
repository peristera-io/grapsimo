pub trait ApproxEq: Copy {
    const TOLERANCE: f64 = 1e-10;

    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool;

    fn approx_eq(self, other: Self) -> bool {
        self.approx_eq_eps(other, Self::TOLERANCE)
    }
}

impl ApproxEq for f64 {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        (self - other).abs() < epsilon
    }
}

#[cfg(test)]
mod tests {
    use crate::ApproxEq;

    #[test]
    pub fn approx_f64() {
        let a = 1.0;
        let b = 1e-100;
        let c = 1e-9;

        assert!((a - b).approx_eq(a));
        assert!(!(a - c).approx_eq(a));
        assert!((a - c).approx_eq_eps(a, 1e-9));
    }
}
