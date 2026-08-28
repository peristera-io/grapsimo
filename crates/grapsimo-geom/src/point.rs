use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::ApproxEq;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl ApproxEq for Point {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        self.x.approx_eq_eps(other.x, epsilon) && self.y.approx_eq_eps(other.y, epsilon)
    }
}

impl Add<Vec2> for Point {
    type Output = Self;

    ///move point by displacement Vec2
    fn add(self, other: Vec2) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point {
    type Output = Vec2;
    /// return the displacement between two points as Vec2
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Sub<Vec2> for Point {
    type Output = Point;

    /// move point by negative vec
    fn sub(self, rhs: Vec2) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// return new vec2
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// return squared length of vec2
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// return length of vec2
    pub fn length(self) -> f64 {
        f64::hypot(self.x, self.y)
    }

    /// return dot product of vec2
    pub fn dot(self, rhs: Vec2) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// return normalized vector, return None if vector length is 0.
    pub fn try_normalize(self) -> Option<Self> {
        match self.length() {
            0.0 => None,
            l => Some(Self {
                x: self.x / l,
                y: self.y / l,
            }),
        }
    }

    /// normalize vector
    /// return NAN vector FOR 0 vectors!
    pub fn normalize(self) -> Self {
        let l = self.length();
        Self {
            x: self.x / l,
            y: self.y / l,
        }
    }

    /// returns the crossproduct with rhs
    pub fn cross(self, rhs: Vec2) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }
}

impl ApproxEq for Vec2 {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        self.x.approx_eq_eps(other.x, epsilon) && self.y.approx_eq_eps(other.y, epsilon)
    }
}

impl Add for Vec2 {
    type Output = Self;
    /// returns the sum of two vec2
    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    /// returns the difference between two vec2
    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    /// returns the scaled vector by f64
    fn mul(self, rhs: f64) -> Self::Output {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;
    /// returns the scaled vector by f64
    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    /// returns the inverse vector
    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl AddAssign for Vec2 {
    /// Add assign Vec2
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Vec2 {
    // Substract assign Vec2
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign<f64> for Vec2 {
    /// multiply assign f64
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
