use crate::{ApproxEq, Point, Vec2};

/// A 2D affine transformation, stored as the six meaningful entries of a
/// 3x3 matrix.
///
/// The full matrix is
///
/// ```text
/// |a  c  e|
/// |b  d  f|
/// |0  0  1|
/// ```
///
/// but the bottom row of an affine matrix is always `[0, 0, 1]`, so only
/// `[a, b, c, d, e, f]` is stored. That layout is column-major: `(a, b)` is
/// the image of the x-axis, `(c, d)` the image of the y-axis, and `(e, f)`
/// the translation. It matches the parameter order of SVG's `matrix(...)`,
/// PostScript, Cairo and CoreGraphics, so [`Affine::to_array`] can be handed
/// to those directly.
///
/// The individual entries are readable via [`a`](Affine::a) ..
/// [`f`](Affine::f).
///
/// # Constructing
///
/// - [`Affine::IDENTITY`] / [`Affine::default`] — leaves everything unchanged
/// - [`Affine::translate`] — move by a [`Vec2`]
/// - [`Affine::rotate`] — rotate about the origin, in radians
/// - [`Affine::rotate_about`] — rotate about an arbitrary [`Point`]
/// - [`Affine::scale`] — uniform scale about the origin
/// - [`Affine::scale_xy`] — independent x/y scale about the origin
/// - [`Affine::scale_about`] / [`Affine::scale_xy_about`] — the same, about
///   an arbitrary [`Point`]
///
/// # Combining and applying
///
/// [`then`](Affine::then) composes in reading order: `a.then(b)` means "do
/// `a`, then do `b`". Note this is the reverse of the usual matrix product
/// notation, where the same transform is written `B * A`.
///
/// [`transform_point`](Affine::transform_point) applies the whole transform.
/// [`transform_vec2`](Affine::transform_vec2) applies only the linear part
/// and ignores the translation, which is what you want for directions,
/// offsets and differences between points.
///
/// # Example
///
/// ```
/// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
///
/// // Scale by 2, then shift right by 10.
/// let t = Affine::scale(2.0).then(Affine::translate(Vec2::new(10.0, 0.0)));
///
/// assert!(t.transform_point(Point::new(3.0, 4.0)).approx_eq(Point::new(16.0, 8.0)));
///
/// // A vector is only scaled — the translation does not apply.
/// assert!(t.transform_vec2(Vec2::new(3.0, 4.0)).approx_eq(Vec2::new(6.0, 8.0)));
/// ```
///
/// # Coordinate system
///
/// All rotation angles go from +x toward +y. In a y-down screen coordinate
/// system that reads as clockwise on screen; in a y-up (mathematical) system
/// it reads as counter-clockwise. The matrix is the same either way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine([f64; 6]);

impl Affine {
    /// The transform that leaves every point and vector unchanged.
    ///
    /// It is also the neutral element of [`then`](Affine::then): composing
    /// anything with it changes nothing.
    ///
    /// ```
    /// use grapsimo_geom::{Affine, Point};
    ///
    /// let p = Point::new(3.0, -4.0);
    /// assert_eq!(Affine::IDENTITY.transform_point(p), p);
    /// assert_eq!(Affine::IDENTITY.to_array(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    /// ```
    pub const IDENTITY: Affine = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// Determinants with a smaller absolute value than this are treated as
    /// singular by [`try_inverse`](Affine::try_inverse).
    ///
    /// This is deliberately tiny rather than a "reasonable" epsilon: it is a
    /// guard against dividing by a denormal or zero, not a judgement about
    /// whether a transform is numerically well-conditioned.
    const SINGULARITY_TOLERANCE: f64 = 1e-100;

    /// Returns the underlying `[a, b, c, d, e, f]` array.
    ///
    /// The order matches SVG's `matrix(a b c d e f)`, Cairo, PostScript and
    /// CoreGraphics, so this is the handoff point to third-party libraries.
    ///
    /// ```
    /// use grapsimo_geom::{Affine, Vec2};
    ///
    /// let t = Affine::translate(Vec2::new(4.0, 5.0));
    /// assert_eq!(t.to_array(), [1.0, 0.0, 0.0, 1.0, 4.0, 5.0]);
    /// ```
    pub fn to_array(self) -> [f64; 6] {
        self.0
    }

    /// Returns the `a` entry: the x-component of the transformed x-axis.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::Affine;
    /// assert_eq!(Affine::scale_xy(2.0, 3.0).a(), 2.0);
    /// ```
    pub fn a(self) -> f64 {
        self.0[0]
    }

    /// Returns the `b` entry: the y-component of the transformed x-axis.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::Affine;
    /// assert_eq!(Affine::scale_xy(2.0, 3.0).b(), 0.0);
    /// ```
    pub fn b(self) -> f64 {
        self.0[1]
    }

    /// Returns the `c` entry: the x-component of the transformed y-axis.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::Affine;
    /// assert_eq!(Affine::scale_xy(2.0, 3.0).c(), 0.0);
    /// ```
    pub fn c(self) -> f64 {
        self.0[2]
    }

    /// Returns the `d` entry: the y-component of the transformed y-axis.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::Affine;
    /// assert_eq!(Affine::scale_xy(2.0, 3.0).d(), 3.0);
    /// ```
    pub fn d(self) -> f64 {
        self.0[3]
    }

    /// Returns the `e` entry, which is the x-translation.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::{Affine, Vec2};
    /// assert_eq!(Affine::translate(Vec2::new(7.0, 9.0)).e(), 7.0);
    /// ```
    pub fn e(self) -> f64 {
        self.0[4]
    }

    /// Returns the `f` entry, which is the y-translation.
    ///
    /// ```text
    /// |a  c  e|
    /// |b  d  f|
    /// |0  0  1|
    /// ```
    ///
    /// ```
    /// # use grapsimo_geom::{Affine, Vec2};
    /// assert_eq!(Affine::translate(Vec2::new(7.0, 9.0)).f(), 9.0);
    /// ```
    pub fn f(self) -> f64 {
        self.0[5]
    }

    /// Returns the transform that moves every point by `translation`.
    ///
    /// Vectors are unaffected — see
    /// [`transform_vec2`](Affine::transform_vec2).
    ///
    /// ```text
    /// |1  0  tx|
    /// |0  1  ty|
    /// |0  0  1 |
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    ///
    /// let t = Affine::translate(Vec2::new(10.0, 22.0));
    /// assert!(t.transform_point(Point::new(-1.0, 3.0)).approx_eq(Point::new(9.0, 25.0)));
    ///
    /// // Translations commute and add.
    /// let u = Affine::translate(Vec2::new(1.0, 2.0));
    /// assert!(t.then(u).approx_eq(u.then(t)));
    /// assert!(t.then(u).approx_eq(Affine::translate(Vec2::new(11.0, 24.0))));
    /// ```
    pub fn translate(translation: Vec2) -> Affine {
        Self([1.0, 0.0, 0.0, 1.0, translation.x, translation.y])
    }

    /// Returns the transform that rotates about the origin by `theta`
    /// radians, from +x toward +y.
    ///
    /// ```text
    /// |cos(θ)  -sin(θ)  0|
    /// |sin(θ)   cos(θ)  0|
    /// |0        0       1|
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    /// use std::f64::consts::PI;
    ///
    /// // 45° takes the unit x-vector onto the diagonal.
    /// let r = Affine::rotate(PI / 4.0);
    /// let d = 2.0f64.sqrt() / 2.0;
    /// assert!(r.transform_point(Point::new(1.0, 0.0)).approx_eq(Point::new(d, d)));
    ///
    /// // A quarter turn maps +x onto +y.
    /// let q = Affine::rotate(PI / 2.0);
    /// assert!(q.transform_point(Point::new(1.0, 0.0)).approx_eq(Point::new(0.0, 1.0)));
    ///
    /// // Four quarter turns are the identity.
    /// assert!(q.then(q).then(q).then(q).approx_eq(Affine::IDENTITY));
    ///
    /// // Rotation preserves length.
    /// let v = Vec2::new(3.0, 2.0);
    /// assert!(q.transform_vec2(v).length().approx_eq(v.length()));
    /// ```
    ///
    /// To rotate about something other than the origin, use
    /// [`rotate_about`](Affine::rotate_about).
    pub fn rotate(theta: f64) -> Affine {
        Self([
            theta.cos(),
            theta.sin(),
            -theta.sin(),
            theta.cos(),
            0.0,
            0.0,
        ])
    }

    /// Returns the transform that rotates about `p` by `theta` radians, from
    /// +x toward +y.
    ///
    /// Equivalent to moving `p` to the origin, rotating, and moving back:
    /// `translate(-p).then(rotate(theta)).then(translate(p))`.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    /// use std::f64::consts::PI;
    ///
    /// let pivot = Point::new(2.0, 0.0);
    /// let r = Affine::rotate_about(pivot, PI);
    ///
    /// // The pivot is the one fixed point.
    /// assert!(r.transform_point(pivot).approx_eq(pivot));
    ///
    /// // A half turn about (2, 0) reflects (0, 0) onto (4, 0).
    /// assert!(r.transform_point(Point::new(0.0, 0.0)).approx_eq(Point::new(4.0, 0.0)));
    /// ```
    pub fn rotate_about(p: Point, theta: f64) -> Affine {
        Affine::translate(-p.to_vec2())
            .then(Affine::rotate(theta))
            .then(Affine::translate(p.to_vec2()))
    }

    /// Returns the transform that scales uniformly about the origin by
    /// `factor`.
    ///
    /// A negative factor mirrors through the origin; `0.0` collapses
    /// everything to a point and produces a transform that cannot be
    /// inverted (see [`try_inverse`](Affine::try_inverse)).
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    ///
    /// let p = Point::new(3.0, -2.0);
    /// assert!(Affine::scale(3.0).transform_point(p).approx_eq(Point::new(9.0, -6.0)));
    /// assert!(Affine::scale(-1.0).transform_point(p).approx_eq(Point::new(-3.0, 2.0)));
    /// assert!(Affine::scale(1.0).approx_eq(Affine::IDENTITY));
    /// ```
    pub fn scale(factor: f64) -> Affine {
        Affine::scale_xy(factor, factor)
    }

    /// Returns the transform that scales uniformly by `factor` about `p`,
    /// leaving `p` fixed.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    ///
    /// let center = Point::new(1.0, 1.0);
    /// let s = Affine::scale_about(center, 2.0);
    ///
    /// assert!(s.transform_point(center).approx_eq(center));
    /// assert!(s.transform_point(Point::new(2.0, 1.0)).approx_eq(Point::new(3.0, 1.0)));
    /// ```
    pub fn scale_about(p: Point, factor: f64) -> Affine {
        Affine::scale_xy_about(p, factor, factor)
    }

    /// Returns the transform that scales about the origin by `factor_x`
    /// along x and `factor_y` along y.
    ///
    /// ```text
    /// |sx  0   0|
    /// |0   sy  0|
    /// |0   0   1|
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    ///
    /// let s = Affine::scale_xy(2.0, 3.0);
    /// assert!(s.transform_point(Point::new(1.0, 1.0)).approx_eq(Point::new(2.0, 3.0)));
    /// assert!(s.transform_point(Point::new(3.0, 2.0)).approx_eq(Point::new(6.0, 6.0)));
    /// ```
    ///
    /// Note that a non-uniform scale does not commute with rotation: scaling
    /// then rotating is not the same as rotating then scaling.
    pub fn scale_xy(factor_x: f64, factor_y: f64) -> Affine {
        Self([factor_x, 0.0, 0.0, factor_y, 0.0, 0.0])
    }

    /// Returns the transform that scales by `factor_x` along x and
    /// `factor_y` along y about `p`, leaving `p` fixed.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    ///
    /// let center = Point::new(10.0, 10.0);
    /// let s = Affine::scale_xy_about(center, 2.0, 0.5);
    ///
    /// assert!(s.transform_point(center).approx_eq(center));
    /// assert!(s.transform_point(Point::new(11.0, 12.0)).approx_eq(Point::new(12.0, 11.0)));
    /// ```
    pub fn scale_xy_about(p: Point, factor_x: f64, factor_y: f64) -> Affine {
        Affine::translate(-p.to_vec2())
            .then(Affine::scale_xy(factor_x, factor_y))
            .then(Affine::translate(p.to_vec2()))
    }

    /// Returns the determinant `a*d - b*c` of the linear part.
    ///
    /// Geometrically this is the signed factor by which the transform
    /// changes area:
    ///
    /// - `1.0`: area preserving (any pure rotation or translation)
    /// - `0.0`: degenerate; the transform collapses the plane onto a line
    ///   or a point and has no inverse
    /// - negative: the transform flips orientation (a mirror)
    ///
    /// The translation does not affect the determinant.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq};
    ///
    /// assert!(Affine::scale_xy(4.0, 15.0).determinant().approx_eq(60.0));
    /// assert!(Affine::rotate(0.51).determinant().approx_eq(1.0));
    /// assert_eq!(Affine::scale(0.0).determinant(), 0.0);
    /// assert!(Affine::scale_xy(-1.0, 1.0).determinant() < 0.0);
    /// ```
    pub fn determinant(self) -> f64 {
        (self.a() * self.d()) - (self.b() * self.c())
    }

    /// Returns the transform that undoes `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    ///
    /// let t = Affine::scale_xy(2.0, 3.0).then(Affine::translate(Vec2::new(5.0, -1.0)));
    /// assert!(t.then(t.inverse()).approx_eq(Affine::IDENTITY));
    ///
    /// let p = Point::new(7.0, 8.0);
    /// assert!(t.inverse().transform_point(t.transform_point(p)).approx_eq(p));
    /// ```
    ///
    /// # Degenerate input
    ///
    /// This divides by the [`determinant`](Affine::determinant) without
    /// checking it. If the determinant is `0.0`, which happens when you
    /// scale by `0.0` on either axis, or apply any other transform that
    /// collapses the plane, the result is filled with `NaN` and `inf`
    /// rather than being reported as an error:
    ///
    /// ```
    /// use grapsimo_geom::Affine;
    ///
    /// assert!(Affine::scale(0.0).inverse().to_array().iter().all(|v| v.is_nan()));
    /// ```
    ///
    /// Use [`try_inverse`](Affine::try_inverse) when the transform may be
    /// degenerate.
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        self.get_inverse(det)
    }

    /// Returns the transform that undoes `self`, or `None` if `self` is
    /// singular.
    ///
    /// "Singular" means the absolute [`determinant`](Affine::determinant) is
    /// at or below `Affine::SINGULARITY_TOLERANCE` (`1e-100`), i.e. the
    /// transform collapses the plane and no inverse exists.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, Point};
    ///
    /// assert!(Affine::scale(2.0).try_inverse().is_some());
    ///
    /// // Collapsing either axis is not invertible.
    /// assert_eq!(Affine::scale(0.0).try_inverse(), None);
    /// assert_eq!(Affine::scale_xy(0.0, 1.0).try_inverse(), None);
    /// assert_eq!(
    ///     Affine::scale_xy_about(Point::new(2.0, 512.0), 0.0, 1.0).try_inverse(),
    ///     None
    /// );
    /// ```
    pub fn try_inverse(self) -> Option<Self> {
        match self.determinant() {
            d if d.abs() > Self::SINGULARITY_TOLERANCE => Some(self.get_inverse(d)),
            _ => None,
        }
    }

    /// Private helper that computes the inverse from an already-known
    /// determinant, so the two public entry points do not recompute it.
    fn get_inverse(self, det: f64) -> Self {
        let Self([a, b, c, d, e, f]) = self;
        Self([
            d / det,
            -b / det,
            -c / det,
            a / det,
            (c * f - d * e) / det,
            (b * e - a * f) / det,
        ])
    }

    /// Composes two transforms: `self.then(other)` is the transform that
    /// applies `self` first and `other` second.
    ///
    /// This is the reverse of matrix-product order. The same composition
    /// written as matrices is `other * self`. Reading `.then()` chains
    /// left-to-right gives the order the transforms actually happen in.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    ///
    /// let t = Affine::translate(Vec2::new(1.0, 0.0));
    /// let s = Affine::scale(2.0);
    /// let p = Point::new(3.0, 0.0);
    ///
    /// // Translate first, then scale: (3,0) -> (4,0) -> (8,0).
    /// assert!(t.then(s).transform_point(p).approx_eq(Point::new(8.0, 0.0)));
    ///
    /// // Scale first, then translate: (3,0) -> (6,0) -> (7,0).
    /// assert!(s.then(t).transform_point(p).approx_eq(Point::new(7.0, 0.0)));
    /// ```
    ///
    /// Composition is associative but not, in general, commutative:
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Vec2};
    /// use std::f64::consts::PI;
    ///
    /// let t = Affine::translate(Vec2::new(2.0, 35.0));
    /// let r = Affine::rotate(PI / 3.0);
    /// let s = Affine::scale_xy(3.5, 67.0);
    ///
    /// assert!(t.then(r.then(s)).approx_eq(t.then(r).then(s)));
    /// assert!(t.then(Affine::IDENTITY).approx_eq(t));
    /// assert!(!t.then(r).approx_eq(r.then(t)));
    /// ```
    pub fn then(self, other: Affine) -> Self {
        let Self([a1, b1, c1, d1, e1, f1]) = self;
        let Self([a2, b2, c2, d2, e2, f2]) = other;
        Self([
            a2 * a1 + c2 * b1,
            b2 * a1 + d2 * b1,
            a2 * c1 + c2 * d1,
            b2 * c1 + d2 * d1,
            a2 * e1 + c2 * f1 + e2,
            b2 * e1 + d2 * f1 + f2,
        ])
    }

    /// Applies the full transform, linear part and translation, to a
    /// point.
    ///
    /// ```text
    /// x' = a*x + c*y + e
    /// y' = b*x + d*y + f
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    ///
    /// let t = Affine::scale(2.0).then(Affine::translate(Vec2::new(1.0, 1.0)));
    /// assert!(t.transform_point(Point::new(3.0, 4.0)).approx_eq(Point::new(7.0, 9.0)));
    /// ```
    pub fn transform_point(self, p: Point) -> Point {
        Point::new(
            self.a() * p.x + self.c() * p.y + self.e(),
            self.b() * p.x + self.d() * p.y + self.f(),
        )
    }

    /// Applies only the linear part of the transform to a vector, ignoring
    /// the translation.
    ///
    /// ```text
    /// x' = a*x + c*y
    /// y' = b*x + d*y
    /// ```
    ///
    /// This is the right operation for anything that represents a direction
    /// or a displacement rather than a location: moving the whole plane must
    /// not change the offset between two of its points.
    ///
    /// # Example
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Vec2};
    ///
    /// let t = Affine::scale(2.0).then(Affine::translate(Vec2::new(100.0, 100.0)));
    ///
    /// // The translation drops out.
    /// assert!(t.transform_vec2(Vec2::new(3.0, 4.0)).approx_eq(Vec2::new(6.0, 8.0)));
    ///
    /// // Which is exactly what makes differences of points consistent.
    /// let p = Point::new(3.0, -42.0);
    /// let q = Point::new(8.0, 57.0);
    /// assert!((t.transform_point(p) - t.transform_point(q)).approx_eq(t.transform_vec2(p - q)));
    /// ```
    pub fn transform_vec2(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.a() * v.x + self.c() * v.y,
            self.b() * v.x + self.d() * v.y,
        )
    }
}

impl Default for Affine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ApproxEq for Affine {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        self.a().approx_eq_eps(other.a(), epsilon)
            && self.b().approx_eq_eps(other.b(), epsilon)
            && self.c().approx_eq_eps(other.c(), epsilon)
            && self.d().approx_eq_eps(other.d(), epsilon)
            && self.e().approx_eq_eps(other.e(), epsilon)
            && self.f().approx_eq_eps(other.f(), epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_approx_eq;
    use std::f64::consts::PI;

    #[test]
    fn identity() {
        let m = Affine::default();

        assert_eq!(m, Affine::IDENTITY);

        let m = Affine::translate(Vec2::new(124.29, 38.1))
            .then(Affine::rotate(0.75).then(Affine::scale_xy(0.3, 12.2)));

        assert_eq!(m.then(Affine::IDENTITY), m);
        assert_eq!(Affine::IDENTITY.then(m), m);

        let m = Affine::IDENTITY;
        let p = Point::new(3.0, 4.12);
        let v = Vec2::new(12.0124, 312.5124);
        assert_eq!(p, m.transform_point(p));
        assert_eq!(v, m.transform_vec2(v));

        assert_eq!(m.to_array(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rotate() {
        let r1 = Affine::rotate(PI / 2.0);
        let r1_array = r1.to_array();
        assert_approx_eq!(r1_array[0], 0.0);
        assert_approx_eq!(r1_array[1], 1.0);
        assert_approx_eq!(r1_array[2], -1.0);
        assert_approx_eq!(r1_array[3], 0.0);
        assert_approx_eq!(r1_array[4], 0.0);
        assert_approx_eq!(r1_array[5], 0.0);

        let r2 = Affine::rotate(-PI / 2.0);
        assert_approx_eq!(r1.then(r2), Affine::IDENTITY);

        let r1 = Affine::rotate(-PI / 3.0);
        let r2 = Affine::rotate(PI / 3.0);
        assert_approx_eq!(r1.then(r2), Affine::IDENTITY);

        let r2 = Affine::rotate(-PI / 2.0);
        assert_approx_eq!(
            r2.transform_point(Point::new(1.0, 0.0)),
            Point::new(0.0, -1.0)
        );

        let r2 = Affine::rotate(PI / 2.0);
        assert_approx_eq!(
            r2.transform_point(Point::new(1.0, 0.0)),
            Point::new(0.0, 1.0)
        );

        let r2 = Affine::rotate(PI / 3.0);
        assert_approx_eq!(
            r2.transform_point(Point::new(1.0, 0.0)),
            Point::new(0.5, 3.0f64.sqrt() / 2.0)
        );

        let v1 = Vec2::new(3.0, 2.0);
        assert_approx_eq!(r2.transform_vec2(v1).length(), v1.length());

        let r1 = Affine::rotate(PI * 2.0);
        assert_approx_eq!(r1, Affine::IDENTITY);

        let r1 = Affine::rotate(PI / 2.0);
        assert_approx_eq!(r1.then(r1).then(r1).then(r1), Affine::IDENTITY);
    }

    #[test]
    fn translate() {
        let v1 = Vec2::new(10.0, 22.0);
        let v2 = Vec2::new(10.0, 9.00);
        assert_approx_eq!(
            Affine::translate(v1).transform_point(Point::new(-1.0, 3.0)),
            Point::new(9.0, 25.0)
        );

        assert_approx_eq!(
            Affine::translate(v2).transform_vec2(Vec2::new(10.0, 12.0)),
            Vec2::new(10.0, 12.0)
        );

        let v3 = v1 + v2;
        let t1 = Affine::translate(v1).then(Affine::translate(v2));
        let t2 = Affine::translate(v2).then(Affine::translate(v1));
        assert_approx_eq!(t1, Affine::translate(v3));
        assert_approx_eq!(t2, Affine::translate(v3));

        let p1 = Point::new(3.01, -42.2);
        let p2 = Point::new(8.31, 57.21);

        assert_approx_eq!(
            t1.transform_point(p1) - t1.transform_point(p2),
            t1.transform_vec2(p1 - p2)
        );
    }

    #[test]
    fn scale() {
        let s1 = Affine::scale(3.0);
        let s2 = Affine::scale(1.0);
        let s3 = Affine::scale(-1.0);

        let p1 = Point::new(3.0, -2.0);

        assert_approx_eq!(s1.transform_point(p1), Point::new(9.0, -6.0));
        assert_approx_eq!(s2.transform_point(p1), p1);
        assert_approx_eq!(s3.transform_point(p1), Point::new(-3.0, 2.0));

        assert_approx_eq!(
            Affine::scale_xy(2.0, 3.0).transform_point(Point::new(1.0, 1.0)),
            Point::new(2.0, 3.0)
        );

        assert_approx_eq!(
            Affine::scale_xy(2.0, 3.0).transform_point(Point::new(3.0, 2.0)),
            Point::new(6.0, 6.0)
        );
    }

    #[test]
    fn combined() {
        let t = Affine::translate(Vec2::new(2.31, 35.86));
        let r = Affine::rotate(PI / 3.0);
        let s = Affine::scale_xy(3.58, 67.12);
        let p = Point::new(2.01, -581.92);
        assert_approx_eq!(
            t.then(r).transform_point(p),
            r.transform_point(t.transform_point(p))
        );
        assert!(!t.then(r).approx_eq(r.then(t)));
        assert_approx_eq!(t.then(r.then(s)), (t.then(r)).then(s));
    }

    #[test]
    fn determinant() {
        let a = Affine::scale(3.214);
        assert_approx_eq!(a.determinant(), 3.214 * 3.214);
        let a = Affine::scale_xy(4.0, 15.0);
        assert_approx_eq!(a.determinant(), 60.0);
        let a = Affine::rotate(0.51);
        assert_approx_eq!(a.determinant(), 1.0);

        let a = Affine::scale(0.0);
        assert_eq!(a.determinant(), 0.0);

        let a = Affine::scale_xy(-1.0, 1.0);
        assert!(a.determinant() < 0.0);
    }

    #[test]
    fn inverse() {
        assert_approx_eq!(
            Affine::scale_xy(2.0, 3.0)
                .inverse()
                .then(Affine::scale_xy(2.0, 3.0)),
            Affine::IDENTITY
        );

        assert_approx_eq!(
            Affine::rotate_about(Point::new(2.51, 95.2), 2.0)
                .inverse()
                .then(Affine::rotate_about(Point::new(2.51, 95.2), 2.0)),
            Affine::IDENTITY
        );

        assert_approx_eq!(
            Affine::translate(Vec2::new(2.51, 95.2))
                .inverse()
                .then(Affine::translate(Vec2::new(2.51, 95.2))),
            Affine::IDENTITY
        );

        let a = Affine::scale_xy_about(Point::new(395.12, 858.0), 2.0, 3.0)
            .then(Affine::rotate_about(Point::new(123.0, -12.0), 0.35))
            .then(Affine::translate(Vec2::new(-3.15, 22.741)));

        assert_approx_eq!(a.then(a.inverse()), Affine::IDENTITY);
        assert_eq!(Affine::scale(0.0).try_inverse(), None);
        assert_eq!(Affine::scale_xy(0.0, 1.0).try_inverse(), None);
        assert_eq!(
            Affine::scale_xy_about(Point::new(2.0, 512.02), 0.0, 1.0).try_inverse(),
            None
        );

        for f in Affine::scale(0.0).inverse().to_array() {
            assert!(f.is_nan());
        }
    }

    #[test]
    fn about_point() {
        let p1 = Point::new(2.1, 58.2);
        let p2 = Point::new(31.56, 581.68);
        let r1 = Affine::rotate_about(p1, 0.0);
        let r2 = Affine::rotate_about(p1, PI);
        let r3 = Affine::rotate_about(p1, PI / 4.0);

        assert_approx_eq!(r1.transform_point(p1), p1);
        assert_approx_eq!(r2.transform_point(p1), p1);
        assert_approx_eq!(r3.transform_point(p1), p1);

        assert_approx_eq!(
            r2,
            Affine::translate(-p1.to_vec2())
                .then(Affine::rotate(PI))
                .then(Affine::translate(p1.to_vec2()))
        );

        assert_approx_eq!((r2.transform_point(p2) - p1).length(), (p2 - p1).length());
    }

    #[test]
    fn approx_eq() {
        let v1 = Vec2::new(0.1, -4.2);
        let v2 = v1 + Vec2::new(9.9e-11, 2e-11);
        assert!(Affine::translate(v1).approx_eq_eps(Affine::translate(v2), 1e-10));
        assert!(!Affine::translate(v1).approx_eq_eps(Affine::translate(v2), 1e-11));
    }
}
