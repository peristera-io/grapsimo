use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::ApproxEq;

/// A location in the 2D plane.
///
/// A `Point` answers *where*, in contrast to [`Vec2`], which answers *how far
/// and in which direction*. They are separate types on purpose: the compiler
/// then rejects operations that have no geometric meaning, such as adding two
/// locations together.
///
/// The operations that are defined form a small, consistent algebra:
///
/// | Expression        | Result   | Meaning                          |
/// |-------------------|----------|----------------------------------|
/// | `point - point`   | [`Vec2`] | displacement from one to another |
/// | `point + vec2`    | `Point`  | move the point                   |
/// | `point - vec2`    | `Point`  | move the point backwards         |
/// | `point + point`   | —        | not defined                      |
///
/// # Example
///
/// ```
/// use grapsimo_geom::{Point, Vec2};
///
/// let a = Point::new(1.0, 1.0);
/// let b = Point::new(4.0, 5.0);
///
/// let d = b - a;
/// assert_eq!(d, Vec2::new(3.0, 4.0));
/// assert_eq!(d.length(), 5.0);
/// assert_eq!(a + d, b);
/// ```
///
/// Use [`ApproxEq`] rather than `==` when the coordinates are the result of
/// arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate. Whether +y points up or down is left to the
    /// consumer of this crate.
    pub y: f64,
}

impl Point {
    /// The point `(0, 0)`.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Vec2};
    /// assert_eq!(Point::ORIGIN + Vec2::new(2.0, 3.0), Point::new(2.0, 3.0));
    /// ```
    pub const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

    /// Creates a point from its coordinates.
    ///
    /// This is a `const fn`, so points can be built in constant context:
    ///
    /// ```
    /// use grapsimo_geom::Point;
    ///
    /// const TOP_LEFT: Point = Point::new(0.0, 0.0);
    /// assert_eq!(TOP_LEFT, Point::ORIGIN);
    /// ```
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Reinterprets the point as the displacement from the origin to it.
    ///
    /// The numbers are unchanged; only the meaning is. This is the escape
    /// hatch for the cases where you genuinely want vector arithmetic on a
    /// location, most commonly to build a transform about a pivot:
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point};
    ///
    /// let pivot = Point::new(3.0, 4.0);
    /// assert_eq!(pivot.to_vec2().length(), 5.0);
    ///
    /// // `translate(-p) then ... then translate(p)` is the about-a-point idiom.
    /// let to_origin = Affine::translate(-pivot.to_vec2());
    /// assert!(to_origin.transform_point(pivot).approx_eq(Point::ORIGIN));
    /// ```
    pub fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

/// Compares both coordinates against the epsilon.
///
/// ```
/// use grapsimo_geom::{ApproxEq, Point};
///
/// let a = Point::new(0.1, 0.2);
/// let b = Point::new(0.3 - 0.2, 0.2);
/// assert!(a != b);
/// assert!(a.approx_eq(b));
/// ```
impl ApproxEq for Point {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        self.x.approx_eq_eps(other.x, epsilon) && self.y.approx_eq_eps(other.y, epsilon)
    }
}

/// Moves a point by a displacement.
///
/// ```
/// use grapsimo_geom::{Point, Vec2};
///
/// assert_eq!(Point::new(1.0, 1.0) + Vec2::new(2.0, 3.0), Point::new(3.0, 4.0));
/// assert_eq!(Point::new(1.0, 1.0) + Vec2::ZERO, Point::new(1.0, 1.0));
/// ```
impl Add<Vec2> for Point {
    type Output = Self;

    fn add(self, other: Vec2) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// Returns the displacement from `rhs` to `self`.
///
/// Read `b - a` as "the vector that takes you from `a` to `b`", so that
/// `a + (b - a) == b`.
///
/// ```
/// use grapsimo_geom::{Point, Vec2};
///
/// let a = Point::new(1.0, 1.0);
/// let b = Point::new(-1.0, -2.0);
///
/// assert_eq!(a - b, Vec2::new(2.0, 3.0));
/// assert_eq!(a - a, Vec2::ZERO);
/// assert_eq!(b + (a - b), a);
/// ```
impl Sub for Point {
    type Output = Vec2;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// Moves a point by the negated displacement.
///
/// ```
/// use grapsimo_geom::{Point, Vec2};
///
/// let p = Point::new(1.0, 1.0);
/// let v = Vec2::new(2.0, 3.0);
/// assert_eq!(p - v, p + (-v));
/// ```
impl Sub<Vec2> for Point {
    type Output = Point;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// A 2D extent: a width and a height.
///
/// `Size` carries no position — it is the shape of a box, not its placement.
/// Pair it with a [`Point`] to get a [`Rect`](crate::Rect).
///
/// Negative components are not rejected, but most consumers treat a size as
/// non-negative; normalise before constructing one if the values come from
/// subtraction.
///
/// ```
/// use grapsimo_geom::{Point, Size};
///
/// let s = Size::new(100.0, 20.0);
/// assert_eq!(s.scaled(2.0, 0.5), Size::new(200.0, 10.0));
///
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    /// Extent along x.
    pub width: f64,
    /// Extent along y.
    pub height: f64,
}

impl Size {
    /// Creates a size from a width and a height.
    ///
    /// `const`, so sizes can be declared as constants.
    ///
    /// ```
    /// use grapsimo_geom::Size;
    ///
    /// const A4_PT: Size = Size::new(595.0, 842.0);
    /// assert_eq!(A4_PT.width, 595.0);
    /// ```
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    /// Reinterprets the extent as a displacement, with `width` becoming `x`
    /// and `height` becoming `y`.
    ///
    /// Useful for offsetting a point by a box's extent, e.g. to get the
    /// far corner of a rectangle:
    ///
    /// ```
    /// use grapsimo_geom::{Point, Size};
    ///
    /// let origin = Point::new(10.0, 10.0);
    /// let size = Size::new(100.0, 20.0);
    /// assert_eq!(origin + size.to_vec2(), Point::new(110.0, 30.0));
    /// ```
    pub fn to_vec2(self) -> Vec2 {
        Vec2 {
            x: self.width,
            y: self.height,
        }
    }

    /// Returns the size with width multiplied by `sx` and height by `sy`.
    ///
    /// This is the extent-only counterpart to
    /// [`Affine::scale_xy`](crate::Affine::scale_xy): it scales the box
    /// without needing a position, which is why it takes two factors rather
    /// than a transform.
    ///
    /// ```
    /// use grapsimo_geom::Size;
    ///
    /// let s = Size::new(4.0, 6.0);
    /// assert_eq!(s.scaled(2.0, 2.0), Size::new(8.0, 12.0));
    /// assert_eq!(s.scaled(1.0, 1.0), s);
    /// assert_eq!(s.scaled(0.5, 1.0 / 3.0), Size::new(2.0, 2.0));
    /// ```
    pub fn scaled(self, sx: f64, sy: f64) -> Self {
        Self {
            width: self.width * sx,
            height: self.height * sy,
        }
    }
}

/// A displacement in the 2D plane: a direction together with a magnitude.
///
/// A `Vec2` is not anchored anywhere. It is what you get by subtracting one
/// [`Point`] from another, and what you add to a `Point` to move it. Unlike a
/// `Point`, a `Vec2` is unaffected by the translation part of an
/// [`Affine`](crate::Affine) — see
/// [`transform_vec2`](crate::Affine::transform_vec2).
///
/// Vectors support the full set of linear operations: addition, subtraction,
/// negation, scaling by a scalar on either side, and the compound assignment
/// forms.
///
/// # Example
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let a = Vec2::new(2.0, 3.0);
/// let b = Vec2::new(-1.0, -2.0);
///
/// assert_eq!(a + b, Vec2::new(1.0, 1.0));
/// assert_eq!(a * 2.0, 2.0 * a);        // scaling works on either side
/// assert_eq!(a - b, a + (-b));
/// assert_eq!(Vec2::new(3.0, 4.0).length(), 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    /// Displacement along x.
    pub x: f64,
    /// Displacement along y.
    pub y: f64,
}

impl Vec2 {
    /// The zero vector: no displacement.
    ///
    /// It is the neutral element of addition, and the one vector with no
    /// direction — [`try_normalize`](Vec2::try_normalize) returns `None` for
    /// it.
    ///
    /// ```
    /// # use grapsimo_geom::Vec2;
    /// let v = Vec2::new(2.0, 3.0);
    /// assert_eq!(v + Vec2::ZERO, v);
    /// assert_eq!(v * 0.0, Vec2::ZERO);
    /// ```
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Creates a vector from its components.
    ///
    /// `const`, so direction constants can be declared:
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// const RIGHT: Vec2 = Vec2::new(1.0, 0.0);
    /// assert_eq!(RIGHT.length(), 1.0);
    /// ```
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns `x² + y²`, the squared length.
    ///
    /// Prefer this over [`length`](Vec2::length) when you only need to
    /// *compare* magnitudes or test against a threshold: it avoids a square
    /// root, and comparing squared lengths gives the same ordering.
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// let a = Vec2::new(3.0, 4.0);
    /// let b = Vec2::new(5.0, 12.0);
    ///
    /// assert_eq!(a.length_squared(), 25.0);
    /// // Same answer as comparing lengths, without the sqrt.
    /// assert_eq!(a.length_squared() < b.length_squared(), a.length() < b.length());
    /// ```
    ///
    /// Note that squaring doubles the exponent, so this overflows to
    /// infinity, or underflows to zero, for magnitudes where
    /// [`length`](Vec2::length) is still fine.
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Returns the Euclidean length (magnitude) of the vector.
    ///
    /// Implemented with [`f64::hypot`], which computes `sqrt(x² + y²)`
    /// without forming `x²` or `y²` directly. That matters at the extremes:
    /// the naive formula overflows to infinity for large components and
    /// flushes to zero for tiny ones, while `hypot` stays accurate.
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// assert_eq!(Vec2::new(4.0, 3.0).length(), 5.0);
    /// assert_eq!(Vec2::ZERO.length(), 0.0);
    ///
    /// // A vector far too small to square without underflowing still has a length.
    /// assert!(Vec2::new(1e-300, 0.0).length() > 0.0);
    /// ```
    ///
    /// The result is never negative.
    pub fn length(self) -> f64 {
        f64::hypot(self.x, self.y)
    }

    /// Returns the dot product `self.x * rhs.x + self.y * rhs.y`.
    ///
    /// Equal to `|self| * |rhs| * cos(θ)`, which makes the sign a cheap test
    /// for relative direction:
    ///
    /// - positive — the vectors point broadly the same way (θ < 90°)
    /// - zero — they are perpendicular
    /// - negative — they point broadly opposite ways (θ > 90°)
    ///
    /// It is commutative, and a vector dotted with itself is its
    /// [`length_squared`](Vec2::length_squared).
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// let right = Vec2::new(1.0, 0.0);
    /// let up = Vec2::new(0.0, 1.0);
    ///
    /// assert_eq!(right.dot(up), 0.0);                  // perpendicular
    /// assert!(right.dot(Vec2::new(0.5, -0.2)) > 0.0);  // same half-plane
    /// assert!(right.dot(Vec2::new(-1.0, -0.3)) < 0.0); // opposite half-plane
    ///
    /// let v = Vec2::new(3.0, 4.0);
    /// assert_eq!(v.dot(v), v.length_squared());
    /// ```
    pub fn dot(self, rhs: Vec2) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Returns the 2D cross product `self.x * rhs.y - self.y * rhs.x`.
    ///
    /// In 2D the cross product is a scalar: the z-component of the 3D cross
    /// product of the two vectors lifted into the plane. It equals
    /// `|self| * |rhs| * sin(θ)`, which is the signed area of the
    /// parallelogram they span. The sign tells you the turn direction from
    /// `self` to `rhs` — which visual direction that corresponds to depends
    /// on whether +y points up or down.
    ///
    /// Zero means the vectors are parallel or antiparallel (collinear),
    /// which is the usual test for "is this point on that line" and for
    /// detecting degenerate segments.
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// let right = Vec2::new(1.0, 0.0);
    /// let up = Vec2::new(0.0, 1.0);
    ///
    /// assert_eq!(right.cross(up), 1.0);
    /// assert_eq!(up.cross(right), -1.0);         // anti-commutative
    /// assert_eq!(right.cross(right), 0.0);       // collinear
    /// assert_eq!(right.cross(right * -3.0), 0.0);
    /// ```
    pub fn cross(self, rhs: Vec2) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Returns the unit vector pointing the same way, or `None` if `self` has
    /// zero length.
    ///
    /// This is the checked counterpart to [`normalize`](Vec2::normalize) and
    /// should be the default choice whenever the vector comes from data —
    /// two coincident points subtract to [`Vec2::ZERO`], and dividing by its
    /// length would silently produce `NaN`.
    ///
    /// ```
    /// use grapsimo_geom::{ApproxEq, Vec2};
    ///
    /// let d = Vec2::new(-62.6, -12.15);
    /// let u = d.try_normalize().unwrap();
    ///
    /// assert!(u.length().approx_eq(1.0));
    /// assert!(u.dot(d) > 0.0);              // same direction...
    /// assert!(u.cross(d).approx_eq(0.0));   // ...and collinear
    ///
    /// assert!(Vec2::ZERO.try_normalize().is_none());
    /// ```
    pub fn try_normalize(self) -> Option<Self> {
        match self.length() {
            0.0 => None,
            l => Some(Self {
                x: self.x / l,
                y: self.y / l,
            }),
        }
    }

    /// Returns the unit vector pointing the same way.
    ///
    /// # Zero vectors
    ///
    /// This divides by [`length`](Vec2::length) without checking it. For
    /// [`Vec2::ZERO`] that is `0.0 / 0.0`, so both components come back as
    /// `NaN` rather than as an error:
    ///
    /// ```
    /// use grapsimo_geom::Vec2;
    ///
    /// let n = Vec2::ZERO.normalize();
    /// assert!(n.x.is_nan() && n.y.is_nan());
    /// ```
    ///
    /// Use [`try_normalize`](Vec2::try_normalize) unless you already know the
    /// vector is non-zero.
    ///
    /// ```
    /// use grapsimo_geom::{ApproxEq, Vec2};
    ///
    /// assert!(Vec2::new(3.0, 4.0).normalize().approx_eq(Vec2::new(0.6, 0.8)));
    ///
    /// // Already-unit vectors are unchanged.
    /// let unit = Vec2::new(1.0, 0.0);
    /// assert!(unit.normalize().approx_eq(unit));
    /// ```
    pub fn normalize(self) -> Self {
        let l = self.length();
        Self {
            x: self.x / l,
            y: self.y / l,
        }
    }
}

/// Compares both components against the epsilon.
///
/// ```
/// use grapsimo_geom::{ApproxEq, Vec2};
///
/// let a = Vec2::new(1.0, 1.0);
/// let b = Vec2::new(1.0 + 1e-11, 1.0);
/// assert!(a != b);
/// assert!(a.approx_eq(b));
/// assert!(!a.approx_eq_eps(b, 1e-12));
/// ```
impl ApproxEq for Vec2 {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        self.x.approx_eq_eps(other.x, epsilon) && self.y.approx_eq_eps(other.y, epsilon)
    }
}

/// Component-wise addition. Commutative and associative.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let a = Vec2::new(2.0, 3.0);
/// let b = Vec2::new(-1.0, -2.0);
/// assert_eq!(a + b, Vec2::new(1.0, 1.0));
/// assert_eq!(a + b, b + a);
/// ```
impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

/// Component-wise subtraction; the same as adding the negation.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let a = Vec2::new(2.0, 3.0);
/// let b = Vec2::new(-1.0, -2.0);
/// assert_eq!(a - b, Vec2::new(3.0, 5.0));
/// assert_eq!(a - b, a + (-b));
/// ```
impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// Scales the vector by a scalar, `vec * factor`.
///
/// A negative factor reverses the direction; `0.0` yields [`Vec2::ZERO`].
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let v = Vec2::new(2.0, 3.0);
/// assert_eq!(v * 2.0, Vec2::new(4.0, 6.0));
/// assert_eq!(v * -1.0, -v);
/// assert_eq!(v * 0.0, Vec2::ZERO);
/// ```
impl Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// Scales the vector by a scalar, `factor * vec`.
///
/// Rust's coherence rules mean `Vec2 * f64` and `f64 * Vec2` are two separate
/// impls; this one exists so scaling reads naturally in either order.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let v = Vec2::new(-1.0, -2.0);
/// assert_eq!(2.0 * v, v * 2.0);
/// ```
impl Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

/// Reverses the direction, keeping the length.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let v = Vec2::new(3.0, -1.0);
/// assert_eq!(-(-v), v);
/// assert_eq!(-Vec2::ZERO, Vec2::ZERO);
/// assert_eq!((-v).length(), v.length());
/// ```
impl Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// In-place addition; equivalent to `a = a + b`.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let mut v = Vec2::new(2.0, 3.0);
/// v += Vec2::new(3.0, -1.0);
/// assert_eq!(v, Vec2::new(5.0, 2.0));
/// ```
impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

/// In-place subtraction; equivalent to `a = a - b`.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let mut v = Vec2::new(2.0, 3.0);
/// v -= Vec2::new(3.0, -1.0);
/// assert_eq!(v, Vec2::new(-1.0, 4.0));
/// ```
impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

/// In-place scaling; equivalent to `v = v * factor`.
///
/// ```
/// use grapsimo_geom::Vec2;
///
/// let mut v = Vec2::new(2.0, 3.0);
/// v *= 3.0;
/// assert_eq!(v, Vec2::new(6.0, 9.0));
/// ```
impl MulAssign<f64> for Vec2 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;

    #[test]
    fn constructors() {
        let p1 = Point::new(256.1, 13.0);
        assert_eq!(p1.x, 256.1);
        assert_eq!(p1.y, 13.0);

        let s1 = Size::new(100.0, 13.2);
        assert_eq!(s1.width, 100.0);
        assert_eq!(s1.height, 13.2);

        let vec2 = Vec2::new(1.0, 2.0);
        assert_eq!(vec2.x, 1.0);
        assert_eq!(vec2.y, 2.0);

        assert_eq!(Vec2::ZERO, Vec2::new(0.0, 0.0));

        const P: Point = Point::new(1.0, 2.0);
        assert_eq!(P.x, 1.0);
        assert_eq!(P.y, 2.0);
    }

    #[test]
    fn point_translation() {
        let p1 = Point::new(1.0, 1.0);
        let p2 = Point::new(-1.0, -2.0);
        let p3 = Point::new(2.0, 3.0);
        let v1 = Vec2::new(2.0, 3.0);
        let v2 = Vec2::new(-1.0, -2.0);

        assert_eq!(p1 - p2, v1);
        assert_eq!(p1 - p3, v2);
        assert_eq!(p1 - p1, Vec2::ZERO);
        assert_eq!(p1 + v2, Point::new(0.0, -1.0));
        assert_eq!(p2 + v1, p1);
        assert_eq!(p1 - v2, p3);
        assert_eq!(p1 + (p1 - p2) - v1, p1);
        assert_eq!(p1 + (p2 - p1), p2);
        assert_eq!(p1 + Vec2::ZERO, p1);
    }

    #[test]
    fn vec_add_sub() {
        let v1 = Vec2::new(2.0, 3.0);
        let v2 = Vec2::new(-1.0, -2.0);
        let v3 = Vec2::new(3.0, -1.0);

        assert_eq!(v1 + v2, Vec2::new(1.0, 1.0));
        assert_eq!(v1 - v2, Vec2::new(3.0, 5.0));
        assert_eq!(v1 + v2, v2 + v1);
        assert_eq!((v1 + v2) + v3, v1 + (v2 + v3));
        assert_eq!(v1 + (-v2), v1 - v2);
        assert_eq!(-(-v1), v1);
        assert_eq!(-Vec2::ZERO, Vec2::ZERO);
    }

    #[test]
    fn vec_scaling() {
        let v1 = Vec2::new(2.0, 3.0);
        let v2 = Vec2::new(-1.0, -2.0);
        let v3 = Vec2::new(3.0, -1.0);

        assert_eq!(v1 * 2.0, Vec2::new(4.0, 6.0));
        assert_eq!(v3 * -1.0, -v3);
        assert_eq!(v1 * 0.0, Vec2::ZERO);

        assert_eq!(v2 * 2.0, 2.0 * v2);
        assert_eq!((v1 + v2) * 2.2, 2.2 * v1 + v2 * 2.2);
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        let v1 = Vec2::new(2.0, 3.0);
        let mut v2 = v1;
        let v3 = Vec2::new(3.0, -1.0);
        v2 += v3;
        assert_eq!(v2, v1 + v3);
        v2 = v1;
        v2 -= v3;
        assert_eq!(v2, v1 - v3);
        v2 = v1;
        v2 *= 3.0;
        assert_eq!(v2, v1 * 3.0);
    }

    #[test]
    fn length() {
        assert_eq!(Vec2::ZERO.length(), 0.0);
        assert_eq!(Vec2::new(4.0, 3.0).length(), 5.0);
        assert_eq!(Vec2::new(5.0, 12.0).length(), 13.0);

        let v1 = Vec2::new(2.3, 4.61);
        assert_approx_eq!(v1.length_squared(), v1.length() * v1.length());

        let v2 = Vec2::new(-1.0, 0.0);
        let v3 = Vec2::new(-62.6, -12.15);
        assert!(v2.length() > 0.0);
        assert!(v3.length() > 0.0);

        assert_approx_eq!((v1 * 3.0).length(), v1.length() * 3.0);
        assert_approx_eq!((v2 * 12.55).length(), v2.length() * 12.55);
    }

    #[test]
    fn dot() {
        let v1 = Vec2::new(1.0, 0.0);
        let v2 = Vec2::new(0.0, 1.0);
        let v3 = Vec2::new(-62.6, -12.15);
        let v4 = Vec2::new(-3.6, 12.15);

        assert!(v1.dot(v2) == 0.0);
        assert!(v2.dot(v1) == 0.0);
        assert_eq!(v3.dot(v4), v4.dot(v3));

        assert!(v3.dot(v3) == v3.length_squared());

        //directionality
        let v1 = Vec2::new(1.0, 0.0);
        let v2 = Vec2::new(0.5, -0.2);
        let v3 = Vec2::new(-1.0, -0.3);

        assert!(v1.dot(v2) > 0.0);
        assert!(v1.dot(v3) < 0.0);
    }

    #[test]
    fn normalize() {
        let v1 = Vec2::new(1.0, 0.0);
        let v2 = Vec2::new(0.0, 1.0);
        let v3 = Vec2::new(-62.6, -12.15);
        let v4 = Vec2::new(-3.6, 12.15);

        // verify normalized vector has unit length
        assert_approx_eq!(
            v1.try_normalize().unwrap().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v2.try_normalize().unwrap().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v1.normalize().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v2.normalize().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v3.try_normalize().unwrap().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v4.try_normalize().unwrap().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v3.normalize().length(),
            1.0,
            "normal vector does not have length 1"
        );
        assert_approx_eq!(
            v4.normalize().length(),
            1.0,
            "normal vector does not have length 1"
        );

        // normalize unit
        assert_approx_eq!(
            v1.normalize(),
            v1,
            "normalized unit vector is not equal to itself"
        );
        assert_approx_eq!(
            v2.normalize(),
            v2,
            "normalized unit vector is not equal to itself"
        );
        assert_approx_eq!(
            v1.try_normalize().unwrap(),
            v1,
            "normalized unit vector is not equal to itself"
        );
        assert_approx_eq!(
            v2.try_normalize().unwrap(),
            v2,
            "normalized unit vector is not equal to itself"
        );

        // verify direction is the same
        assert!(v1.normalize().dot(v1) > 0.0);
        assert_approx_eq!(
            v1.normalize().cross(v1),
            0.0,
            "normalized vector has not the same direction"
        );
        assert!(v1.try_normalize().unwrap().dot(v1) > 0.0);
        assert_approx_eq!(
            v1.try_normalize().unwrap().cross(v1),
            0.0,
            "normalized vector has not the same direction"
        );
        assert!(v3.try_normalize().unwrap().dot(v3) > 0.0);
        assert_approx_eq!(
            v3.try_normalize().unwrap().cross(v3),
            0.0,
            "normalized vector has not the same direction"
        );
        assert!(v3.normalize().dot(v3) > 0.0);
        assert_approx_eq!(
            v3.normalize().cross(v3),
            0.0,
            "normalized vector has not the same direction"
        );
    }

    #[test]
    fn normalize_of_zero() {
        assert!(Vec2::ZERO.try_normalize().is_none());
        assert!(Vec2::ZERO.normalize().x.is_nan());
        assert!(Vec2::ZERO.normalize().y.is_nan());
    }

    #[test]
    fn normalize_of_tiny_vector() {
        let v1 = Vec2::new(1e-300, 0.0);
        assert!(
            v1.normalize().length_squared() > 0.0,
            "Length should not be 0 for tiny vector"
        );
    }
}
