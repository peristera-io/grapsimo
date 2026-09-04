#![warn(missing_docs)]

//! 2D geometry primitives for text layout and rendering.
//!
//! This crate is the foundation layer: pure `f64` value types with no
//! rendering, no allocation and no dependencies. Everything is `Copy`, and
//! every operation takes and returns values rather than mutating in place
//! (with the exception of the `*Assign` operators on [`Vec2`]).
//!
//! # The types
//!
//! - [`Point`] — a location in the plane
//! - [`Vec2`] — a displacement or direction; the difference between two
//!   [`Point`]s
//! - [`Size`] — a width/height extent
//! - [`Rect`] — an axis-aligned rectangle
//! - [`Affine`] — a 2D affine transformation
//! - [`ApproxEq`] — tolerance-based comparison, since exact `f64` equality is
//!   rarely what you want after arithmetic
//!
//! # Point vs. Vec2
//!
//! [`Point`] and [`Vec2`] are both "two f64s", but they are deliberately
//! distinct types because they behave differently. A point is *where*
//! something is; a vector is *how far and in which direction*. The type
//! system encodes which operations make sense:
//!
//! ```
//! use grapsimo_geom::{Point, Vec2};
//!
//! let a = Point::new(1.0, 1.0);
//! let b = Point::new(4.0, 5.0);
//!
//! let d: Vec2 = b - a;          // point - point = displacement
//! assert_eq!(d, Vec2::new(3.0, 4.0));
//!
//! let c: Point = a + d;         // point + displacement = point
//! assert_eq!(c, b);
//! ```
//!
//! Adding two [`Point`]s is not defined, because the sum of two locations has
//! no meaning. The same distinction shows up in [`Affine`]: translating a
//! point moves it, while translating a vector leaves it unchanged — see
//! [`Affine::transform_point`] and [`Affine::transform_vec2`].
//!
//! # Coordinate system
//!
//! The crate does not impose an axis direction: +y may point up or down
//! depending on the consumer. Rotations always go from +x toward +y, which
//! reads as clockwise on a y-down screen and counter-clockwise in a
//! conventional y-up system.

mod point;
pub use point::{Point, Size, Vec2};

mod approx;
pub use approx::ApproxEq;

mod rect;
pub use rect::Rect;

mod affine;
pub use affine::Affine;

#[cfg(test)]
mod test_util;
