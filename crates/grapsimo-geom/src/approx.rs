/// Tolerance-based equality for floating-point types.
///
/// Exact `==` on `f64` is almost never the right test after arithmetic:
/// rounding makes results that are mathematically equal differ in the last
/// few bits. This trait provides "equal to within a tolerance" instead, and
/// is implemented for [`f64`] and for every geometry type in this crate.
///
/// ```
/// use grapsimo_geom::ApproxEq;
///
/// // Exact equality fails here; approximate equality does not.
/// let sum = 0.1 + 0.2;
/// assert!(sum != 0.3);
/// assert!(sum.approx_eq(0.3));
/// ```
///
/// # Implementing
///
/// Only [`approx_eq_eps`](ApproxEq::approx_eq_eps) has to be written;
/// [`approx_eq`](ApproxEq::approx_eq) is provided and simply forwards the
/// type's [`TOLERANCE`](ApproxEq::TOLERANCE). Composite types compare
/// component by component, passing the caller's epsilon down unchanged:
///
/// ```
/// use grapsimo_geom::ApproxEq;
///
/// #[derive(Clone, Copy)]
/// struct Pair(f64, f64);
///
/// impl ApproxEq for Pair {
///     fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
///         self.0.approx_eq_eps(other.0, epsilon) && self.1.approx_eq_eps(other.1, epsilon)
///     }
/// }
///
/// assert!(Pair(1.0, 2.0).approx_eq(Pair(1.0, 2.0 + 1e-15)));
/// ```
///
/// # Caveats
///
/// The comparison is a fixed *absolute* difference, not a relative one. That
/// is the right choice for coordinates in a bounded space such as a layout or
/// a page, where magnitudes stay within a predictable range. It is the wrong
/// choice for values spanning many orders of magnitude: at `1e12` the gap
/// between neighbouring `f64` values already exceeds the default tolerance,
/// so two genuinely distinct numbers can compare equal, and at `1e-30`
/// everything compares equal to zero.
///
/// The relation is also not transitive — `a.approx_eq(b)` and
/// `b.approx_eq(c)` do not imply `a.approx_eq(c)` — so it must not be used to
/// key a `HashMap`, sort, or stand in for `PartialEq` in any algorithm that
/// assumes an equivalence relation. It is a testing and tolerance-checking
/// tool.
pub trait ApproxEq: Copy {
    /// The default epsilon used by [`approx_eq`](ApproxEq::approx_eq).
    ///
    /// `1e-10` is comfortably above the rounding error accumulated by the
    /// handful of operations in a typical transform chain, and comfortably
    /// below any distance that matters at layout scale.
    ///
    /// Implementors may override it; pass an explicit epsilon to
    /// [`approx_eq_eps`](ApproxEq::approx_eq_eps) to override it per call.
    const TOLERANCE: f64 = 1e-10;

    /// Returns `true` if `self` and `other` differ by strictly less than
    /// `epsilon`.
    ///
    /// For composite types every component must satisfy the bound.
    ///
    /// ```
    /// use grapsimo_geom::ApproxEq;
    ///
    /// assert!(1.0.approx_eq_eps(1.5, 1.0));
    /// assert!(!1.0.approx_eq_eps(1.5, 0.5)); // strict: 0.5 < 0.5 is false
    /// ```
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool;

    /// Returns `true` if `self` and `other` differ by less than
    /// [`TOLERANCE`](ApproxEq::TOLERANCE).
    ///
    /// ```
    /// use grapsimo_geom::{ApproxEq, Point, Vec2};
    ///
    /// assert!((1.0f64 + 1e-11).approx_eq(1.0));
    /// assert!(!(1.0f64 + 1e-9).approx_eq(1.0));
    ///
    /// // Available on the geometry types too.
    /// assert!(Point::new(1.0, 2.0).approx_eq(Point::new(1.0, 2.0 + 1e-12)));
    /// assert!(Vec2::new(3.0, 4.0).normalize().length().approx_eq(1.0));
    /// ```
    fn approx_eq(self, other: Self) -> bool {
        self.approx_eq_eps(other, Self::TOLERANCE)
    }
}

/// Compares two `f64` by absolute difference.
///
/// `NaN` is never approximately equal to anything, including itself, because
/// `NaN - x` is `NaN` and every comparison against `NaN` is false. Two
/// infinities of the same sign are also *not* approximately equal, since
/// `inf - inf` is `NaN`.
///
/// ```
/// use grapsimo_geom::ApproxEq;
///
/// assert!(!f64::NAN.approx_eq(f64::NAN));
/// assert!(!f64::INFINITY.approx_eq(f64::INFINITY));
/// ```
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
        let b = 1e-11;
        let c = 1e-9;

        assert!((a + b).approx_eq(a));
        assert!(!(a + b).approx_eq_eps(a, 1e-11));
        assert!(!(a - c).approx_eq(a));
        assert!((a - c).approx_eq_eps(a, 1e-9));
    }
}
