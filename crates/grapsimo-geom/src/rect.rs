use crate::{Affine, ApproxEq, Point, Size, Vec2};

/// An axis-aligned rectangle.
///
/// Stored as two opposing corners rather than an origin and a [`Size`], which
/// makes the common queries (`x0`/`x1`, intersection, union, containment)
/// direct comparisons with no arithmetic.
///
/// # Invariant
///
/// Every constructor normalises, so `min` always holds the lower x and the
/// lower y and `max` the higher of each. The corners are private precisely to
/// keep that guarantee: you cannot build an inverted `Rect`, and code
/// downstream never has to defensively sort coordinates.
///
/// ```
/// use grapsimo_geom::{Point, Rect};
///
/// // Corners given in "wrong" order — still normalised.
/// let r = Rect::from_points(Point::new(10.0, 10.0), Point::new(0.0, 0.0));
/// assert_eq!(r.min(), Point::new(0.0, 0.0));
/// assert_eq!(r.max(), Point::new(10.0, 10.0));
/// assert_eq!(r.width(), 10.0);
/// ```
///
/// # Axis orientation
///
/// "min" and "max" are about coordinate values, not about screen position.
/// With +y pointing down, `min` is the visually top-left corner; with +y
/// pointing up it is the bottom-left one. The arithmetic is identical either
/// way; only the names you would give the corners change.
///
/// # Fallible operations
///
/// There is no "no rectangle" value. An operation that cannot produce a
/// rectangle says so in its return type instead:
///
/// | Returns `Option<Rect>` | because |
/// |---|---|
/// | [`intersect`](Rect::intersect) | the rectangles may not overlap |
/// | [`inflate`](Rect::inflate) | a shrink may over-shoot the centre |
/// | [`round_in`](Rect::round_in) | no integer-aligned box may fit inside |
/// | [`bounding`](Rect::bounding) | the iterator may be empty |
///
/// Everything else — [`union`](Rect::union),
/// [`union_point`](Rect::union_point), [`translate`](Rect::translate),
/// [`round`](Rect::round), [`round_out`](Rect::round_out),
/// [`transform_bounds`](Rect::transform_bounds) — always yields a rectangle.
///
/// The trade-off is that `Rect` has no neutral element, so you cannot seed a
/// fold with one. Accumulate through `Option<Rect>` instead: `reduce` for
/// rectangles, [`bounding`](Rect::bounding) for points.
///
/// # Emptiness
///
/// Separately from the above, a rectangle whose extent collapses on either
/// axis is *empty* — see [`is_empty`](Rect::is_empty). This is a property of
/// a rectangle that exists, not a failure: `from_points(p, p)` is a perfectly
/// good value that happens to enclose nothing.
///
/// # Example
///
/// ```
/// use grapsimo_geom::{Point, Rect, Size};
///
/// let a = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
/// let b = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
///
/// assert!(a.intersects(b));
/// assert_eq!(a.intersect(b).map(Rect::size), Some(Size::new(5.0, 5.0)));
/// assert_eq!(a.union(b).size(), Size::new(15.0, 15.0));
/// assert!(a.contains(Point::new(1.0, 1.0)));
///
/// // Accumulating over a collection: `reduce` gives you the Option.
/// let far = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
/// let bounds = [a, b, far].into_iter().reduce(Rect::union);
/// assert_eq!(bounds.map(Rect::max), Some(Point::new(21.0, 21.0)));
/// assert_eq!([].into_iter().reduce(Rect::union), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    /// Returns the rectangle spanned by two opposing corners, in any order.
    ///
    /// The corners are sorted per axis, so `from_points(a, b)` and
    /// `from_points(b, a)` are the same rectangle. This is the constructor to
    /// reach for when both corners come from data — a drag gesture, a pair of
    /// glyph positions — and neither is known to be the smaller.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let a = Point::new(-28.0, 14.0);
    /// let b = Point::new(28.0, -14.0);
    ///
    /// let r = Rect::from_points(a, b);
    /// assert_eq!(r, Rect::from_points(b, a));
    /// assert_eq!(r.min(), Point::new(-28.0, -14.0));
    /// assert_eq!(r.max(), Point::new(28.0, 14.0));
    /// assert_eq!(r.center(), Point::ORIGIN);
    /// ```
    pub const fn from_points(a: Point, b: Point) -> Rect {
        Self {
            min: Point::new(a.x.min(b.x), a.y.min(b.y)),
            max: Point::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }

    /// Returns the rectangle with one corner at `origin` and the given
    /// extent.
    ///
    /// A negative width or height is allowed: the rectangle extends in the
    /// negative direction and is then normalised, so `origin` ends up as the
    /// `max` corner on that axis rather than the `min` one.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::new(10.0, 10.0), Size::new(5.0, 5.0));
    /// assert_eq!(r.min(), Point::new(10.0, 10.0));
    /// assert_eq!(r.max(), Point::new(15.0, 15.0));
    ///
    /// // Negative extent grows backwards from the origin.
    /// let r = Rect::from_origin_size(Point::new(10.0, 10.0), Size::new(-5.0, -5.0));
    /// assert_eq!(r.min(), Point::new(5.0, 5.0));
    /// assert_eq!(r.max(), Point::new(10.0, 10.0));
    /// ```
    pub const fn from_origin_size(origin: Point, size: Size) -> Self {
        let x = origin.x + size.width;
        let y = origin.y + size.height;

        Self {
            min: Point::new(origin.x.min(x), origin.y.min(y)),
            max: Point::new(origin.x.max(x), origin.y.max(y)),
        }
    }

    /// Returns the rectangle of the given extent centred on `center`.
    ///
    /// The size is taken as an absolute magnitude, so a negative width or
    /// height gives the same rectangle as its positive counterpart — unlike
    /// [`from_origin_size`](Rect::from_origin_size), where the sign picks a
    /// direction.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let c = Point::new(1.0, 1.0);
    /// let r = Rect::from_center_size(c, Size::new(2.0, 2.0));
    ///
    /// assert_eq!(r.center(), c);
    /// assert_eq!(r, Rect::from_points(Point::ORIGIN, Point::new(2.0, 2.0)));
    ///
    /// // Sign of the size is irrelevant here.
    /// assert_eq!(r, Rect::from_center_size(c, Size::new(-2.0, -2.0)));
    /// ```
    pub const fn from_center_size(center: Point, size: Size) -> Self {
        Self {
            min: Point::new(
                center.x - (size.width.abs() * 0.5),
                center.y - (size.height.abs() * 0.5),
            ),
            max: Point::new(
                center.x + (size.width.abs() * 0.5),
                center.y + (size.height.abs() * 0.5),
            ),
        }
    }

    /// Returns the corner with the lower x and lower y coordinate.
    ///
    /// Same as [`origin`](Rect::origin).
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.min(), Point::new(0.0, 1.0));
    /// ```
    pub fn min(self) -> Point {
        self.min
    }

    /// Returns the corner with the higher x and higher y coordinate.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.max(), Point::new(4.0, 3.0));
    /// ```
    pub fn max(self) -> Point {
        self.max
    }

    /// Returns the lower x coordinate (the left edge).
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// # let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.x0(), 0.0);
    /// ```
    pub fn x0(self) -> f64 {
        self.min.x
    }

    /// Returns the higher x coordinate (the right edge).
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// # let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.x1(), 4.0);
    /// ```
    pub fn x1(self) -> f64 {
        self.max.x
    }

    /// Returns the lower y coordinate.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// # let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.y0(), 1.0);
    /// ```
    pub fn y0(self) -> f64 {
        self.min.y
    }

    /// Returns the higher y coordinate.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// # let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.y1(), 3.0);
    /// ```
    pub fn y1(self) -> f64 {
        self.max.y
    }

    /// Returns the origin: the corner with the minimal x and y coordinate.
    ///
    /// Together with [`size`](Rect::size) this round-trips through
    /// [`from_origin_size`](Rect::from_origin_size).
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(Rect::from_origin_size(r.origin(), r.size()), r);
    /// ```
    pub fn origin(self) -> Point {
        self.min
    }

    /// Returns the extent along x. Never negative, by the normalisation
    /// invariant.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.width(), 4.0);
    /// ```
    pub fn width(self) -> f64 {
        self.x1() - self.x0()
    }

    /// Returns the extent along y. Never negative, by the normalisation
    /// invariant.
    ///
    /// ```
    /// # use grapsimo_geom::{Point, Rect};
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.height(), 2.0);
    /// ```
    pub fn height(self) -> f64 {
        self.y1() - self.y0()
    }

    /// Returns the extent as a [`Size`], dropping the position.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(r.size(), Size::new(4.0, 2.0));
    /// ```
    pub fn size(self) -> Size {
        Size::new(self.width(), self.height())
    }

    /// Returns the midpoint of the rectangle.
    ///
    /// Computed as the average of the two edges per axis rather than
    /// `min + size/2`, which keeps it symmetric and well behaved for
    /// rectangles far from the origin.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let c = Point::new(3.0, -7.0);
    /// let r = Rect::from_center_size(c, Size::new(4.0, 2.0));
    /// assert_eq!(r.center(), c);
    /// ```
    pub fn center(self) -> Point {
        Point::new((self.x0() + self.x1()) * 0.5, (self.y0() + self.y1()) * 0.5)
    }

    /// Returns `width * height`. Zero for an empty rectangle.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.5));
    /// assert_eq!(r.area(), 10.0);
    /// let r = Rect::from_points(Point::ORIGIN, Point::ORIGIN);
    /// assert_eq!(r.area(), 0.0);
    /// ```
    pub fn area(self) -> f64 {
        self.width() * self.height()
    }

    /// Returns the four corners in winding order, starting at
    /// [`min`](Rect::min):
    ///
    /// `[(x0,y0), (x0,y1), (x1,y1), (x1,y0)]`
    ///
    /// Which visual corners those are depends on the axis orientation. With
    /// +y up the order reads `[lower-left, upper-left, upper-right,
    /// lower-right]`; with +y down it reads `[upper-left, lower-left,
    /// lower-right, upper-right]`. Either way consecutive entries share an
    /// edge, which is what matters when feeding a path or a transform.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(2.0, 1.0));
    /// assert_eq!(
    ///     r.corners(),
    ///     [
    ///         Point::new(0.0, 0.0),
    ///         Point::new(0.0, 1.0),
    ///         Point::new(2.0, 1.0),
    ///         Point::new(2.0, 0.0),
    ///     ]
    /// );
    /// ```
    pub fn corners(self) -> [Point; 4] {
        [
            self.min(),
            Point::new(self.x0(), self.y1()),
            self.max(),
            Point::new(self.x1(), self.y0()),
        ]
    }

    /// Returns `true` if the rectangle has no interior — it has collapsed to
    /// a line or a point on at least one axis. The area is 0.
    ///
    /// A zero-width or zero-height rectangle returns true, because
    /// containment is half-open (see [`contains`](Rect::contains)): a
    /// degenerate rectangle contains no points at all, so treating it as
    /// non-empty would be inconsistent. Nothing
    /// [`intersects`](Rect::intersects) an empty rectangle either.
    ///
    /// The operations that *could* produce an empty rectangle return
    /// `Option` instead, so an empty one only ever arrives from a
    /// constructor or from an operation that has no failure case:
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// assert!(!Rect::from_origin_size(Point::ORIGIN, Size::new(1.0, 1.0)).is_empty());
    ///
    /// // Zero extent, straight from a constructor.
    /// assert!(Rect::from_origin_size(Point::ORIGIN, Size::new(1.0, 0.0)).is_empty());
    /// assert!(Rect::from_points(Point::new(3.0, 4.0), Point::new(3.0, 4.0)).is_empty());
    ///
    /// // A single point has a position but no area.
    /// assert!(Rect::bounding([Point::new(3.0, 4.0)]).unwrap().is_empty());
    ///
    /// // `round` has no failure case, so a sliver collapses in place —
    /// // unlike `round_in`, which returns None.
    /// let sliver = Rect::from_points(Point::new(1.1, 0.0), Point::new(1.2, 4.0));
    /// assert!(sliver.round().is_empty());
    /// assert!(sliver.round_in().is_none());
    /// ```
    pub fn is_empty(self) -> bool {
        self.x0() >= self.x1() || self.y0() >= self.y1()
    }

    /// Returns `true` if `point` lies inside the rectangle.
    ///
    /// The bounds are **half-open**: the `min` edges are inclusive and the
    /// `max` edges exclusive, i.e. `[x0, x1) × [y0, y1)`. This is the usual
    /// convention for pixel coverage, and it means adjacent rectangles tile
    /// the plane without any point being claimed twice.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(2.0, 2.0));
    ///
    /// assert!(r.contains(Point::new(1.0, 1.0)));
    /// assert!(r.contains(Point::new(0.0, 0.0)));   // min corner: inside
    /// assert!(!r.contains(Point::new(2.0, 2.0)));  // max corner: outside
    /// assert!(!r.contains(Point::new(2.0, 1.0)));  // max edge: outside
    ///
    /// // Consequently an empty rectangle contains nothing.
    /// assert!(!Rect::from_points(Point::ORIGIN, Point::ORIGIN).contains(Point::ORIGIN));
    /// ```
    pub fn contains(self, point: Point) -> bool {
        (point.x >= self.x0())
            && (point.x < self.x1())
            && (point.y >= self.y0())
            && (point.y < self.y1())
    }

    /// Returns `true` if `other` lies entirely within `self`.
    ///
    /// Unlike [`contains`](Rect::contains) this comparison is **closed** on
    /// both ends: a rectangle flush against an edge still counts as
    /// contained, and every rectangle contains itself.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let outer = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
    /// let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));
    /// let flush = Rect::from_origin_size(Point::new(8.0, 8.0), Size::new(2.0, 2.0));
    ///
    /// assert!(outer.contains_rect(inner));
    /// assert!(outer.contains_rect(flush));  // touching the max edge still counts
    /// assert!(outer.contains_rect(outer));
    /// assert!(!inner.contains_rect(outer));
    /// ```
    pub fn contains_rect(self, other: Rect) -> bool {
        (other.x0() >= self.x0())
            && (other.x1() <= self.x1())
            && (other.y0() >= self.y0())
            && (other.y1() <= self.y1())
    }

    /// Returns `true` if the two rectangles share any interior area.
    ///
    /// Rectangles that merely touch along an edge do **not** intersect —
    /// their overlap has zero area. This is exactly the condition under
    /// which [`intersect`](Rect::intersect) returns a non-empty rectangle.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let a = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
    /// let overlapping = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
    /// let touching = Rect::from_origin_size(Point::new(10.0, 0.0), Size::new(5.0, 5.0));
    /// let disjoint = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
    ///
    /// assert!(a.intersects(overlapping));
    /// assert!(!a.intersects(touching));
    /// assert!(!a.intersects(disjoint));
    /// ```
    pub fn intersects(self, other: Rect) -> bool {
        if self.is_empty() || other.is_empty() {
            false
        } else {
            (self.x0() < other.x1())
                && (other.x0() < self.x1())
                && (self.y0() < other.y1())
                && (other.y0() < self.y1())
        }
    }

    /// Returns the overlapping region of the two rectangles, or
    /// None if they do not overlap.
    ///
    /// This is the operation for clipping: intersecting a shape's bounds with
    /// a viewport gives the region actually worth drawing.
    ///
    /// An empty operand always gives `None`, which is exactly the condition
    /// [`intersects`](Rect::intersects) reports — the two agree in every
    /// case, so `a.intersects(b) == a.intersect(b).is_some()`.
    ///
    /// To test a single point rather than a region, use
    /// [`contains`](Rect::contains).
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let a = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
    /// let b = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
    ///
    /// assert_eq!(
    ///     a.intersect(b).unwrap(),
    ///     Rect::from_points(Point::new(5.0, 5.0), Point::new(10.0, 10.0))
    /// );
    ///
    /// // Containment: the intersection is the smaller rectangle.
    /// let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));
    /// assert_eq!(a.intersect(inner).unwrap(), inner);
    ///
    /// // Idempotent, commutative.
    /// assert_eq!(a.intersect(a).unwrap(), a);
    /// assert_eq!(a.intersect(b).unwrap(), b.intersect(a).unwrap());
    ///
    /// // Disjoint and edge-touching both give None.
    /// let far = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
    /// let touching = Rect::from_origin_size(Point::new(10.0, 0.0), Size::new(5.0, 5.0));
    /// assert!(a.intersect(far).is_none());
    /// assert!(a.intersect(touching).is_none());
    /// ```
    ///
    /// Clipping reads naturally, because `None` is precisely "nothing to
    /// draw":
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let viewport = Rect::from_origin_size(Point::ORIGIN, Size::new(800.0, 600.0));
    /// let glyph = Rect::from_origin_size(Point::new(790.0, 10.0), Size::new(20.0, 20.0));
    ///
    /// let Some(visible) = viewport.intersect(glyph) else {
    ///     unreachable!("the glyph straddles the right edge")
    /// };
    /// assert_eq!(visible.size(), Size::new(10.0, 20.0));
    /// ```
    pub fn intersect(self, other: Rect) -> Option<Rect> {
        let x_min = self.x0().max(other.x0());
        let x_max = self.x1().min(other.x1());
        let y_min = self.y0().max(other.y0());
        let y_max = self.y1().min(other.y1());

        if x_min < x_max && y_min < y_max {
            Some(Rect::from_points(
                Point::new(x_min, y_min),
                Point::new(x_max, y_max),
            ))
        } else {
            None
        }
    }

    /// Returns the smallest rectangle containing both `self` and `other`.
    ///
    /// This is a bounding box, not a set union: the result covers the gap
    /// between two disjoint rectangles as well.
    ///
    /// There are no special cases: every operand contributes its corners,
    /// including an empty one. That means `union` has no identity element, so
    /// there is no value to seed a fold with — use `reduce`, which returns
    /// `None` for an empty sequence, or [`bounding`](Rect::bounding) when you
    /// are starting from points.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let a = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
    /// let b = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
    ///
    /// assert_eq!(
    ///     a.union(b),
    ///     Rect::from_points(Point::ORIGIN, Point::new(15.0, 15.0))
    /// );
    ///
    /// // Disjoint inputs: the gap is covered too.
    /// let far = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
    /// assert_eq!(a.union(far).max(), Point::new(21.0, 21.0));
    ///
    /// // An empty rect still contributes its corners; it is not skipped.
    /// let r = Rect::from_points(Point::new(-1.0, -20.0), Point::new(-1.0, -10.0));
    /// assert!(r.is_empty());
    /// assert_eq!(r.union(a), Rect::from_points(Point::new(-1.0, -20.0), Point::new(10.0, 10.0)));
    ///
    /// // Commutative, idempotent, and associative.
    /// assert_eq!(a.union(b), b.union(a));
    /// assert_eq!(a.union(a), a);
    /// assert_eq!(a.union(b).union(far), a.union(b.union(far)));
    ///
    /// // Accumulating over a sequence.
    /// assert_eq!([a, b, far].into_iter().reduce(Rect::union), Some(a.union(b).union(far)));
    /// ```
    pub fn union(self, other: Rect) -> Rect {
        let x_min = self.x0().min(other.x0());
        let x_max = self.x1().max(other.x1());
        let y_min = self.y0().min(other.y0());
        let y_max = self.y1().max(other.y1());
        Rect::from_points(Point::new(x_min, y_min), Point::new(x_max, y_max))
    }

    /// Returns the smallest rectangle containing both `self` and `point`.
    ///
    /// This expands an existing rectangle to reach `point`. It always
    /// succeeds, because there is already a rectangle to grow. To build
    /// bounds from scratch, where there may be no points at all, use
    /// [`bounding`](Rect::bounding).
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let start = Rect::from_points(Point::new(3.0, 4.0), Point::new(4.0, 5.0));
    /// let bounds = [Point::new(-1.0, 9.0), Point::new(5.0, 0.0)]
    ///     .into_iter()
    ///     .fold(start, Rect::union_point);
    ///
    /// assert_eq!(bounds.min(), Point::new(-1.0, 0.0));
    /// assert_eq!(bounds.max(), Point::new(5.0, 9.0));
    /// ```
    pub fn union_point(self, point: Point) -> Rect {
        let x_min = self.x0().min(point.x);
        let x_max = self.x1().max(point.x);
        let y_min = self.y0().min(point.y);
        let y_max = self.y1().max(point.y);

        Rect::from_points(Point::new(x_min, y_min), Point::new(x_max, y_max))
    }

    /// Returns the smallest rectangle containing all `points`, or `None` if
    /// the iterator is empty.
    ///
    /// This is the entry point for building bounds from scratch: it consumes
    /// the first point to seed the rectangle and folds
    /// [`union_point`](Rect::union_point) over the rest, so no neutral
    /// starting value is needed.
    ///
    /// Note that `Some` does not imply a rectangle with area — a single point
    /// yields a degenerate one, for which [`is_empty`](Rect::is_empty) is
    /// true.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let bounds = Rect::bounding([
    ///     Point::new(3.0, 4.0),
    ///     Point::new(-1.0, 9.0),
    ///     Point::new(5.0, 0.0),
    /// ]).unwrap();
    ///
    /// assert_eq!(bounds.min(), Point::new(-1.0, 0.0));
    /// assert_eq!(bounds.max(), Point::new(5.0, 9.0));
    ///
    /// // No points at all, and a single point.
    /// assert_eq!(Rect::bounding([]), None);
    /// assert!(Rect::bounding([Point::new(3.0, 4.0)]).unwrap().is_empty());
    ///
    /// // Accepts anything iterable over points, and round-trips a rect's
    /// // own corners.
    /// let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
    /// assert_eq!(Rect::bounding(r.corners()), Some(r));
    /// ```
    pub fn bounding(points: impl IntoIterator<Item = Point>) -> Option<Rect> {
        let mut points = points.into_iter();
        points
            .next()
            .map(|first| points.fold(Rect::from_points(first, first), Rect::union_point))
    }

    /// Returns the rectangle moved by `offset`, with the size unchanged.
    ///
    /// The pure-translation shortcut for
    /// [`transform_bounds`](Rect::transform_bounds), and exact rather than
    /// conservative because translation cannot rotate the box.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size, Vec2};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.0));
    /// let moved = r.translate(Vec2::new(10.0, -5.0));
    ///
    /// assert_eq!(moved.origin(), Point::new(10.0, -5.0));
    /// assert_eq!(moved.size(), r.size());
    /// assert_eq!(moved.translate(Vec2::new(-10.0, 5.0)), r);
    /// ```
    pub fn translate(self, offset: Vec2) -> Rect {
        Rect::from_points(self.min + offset, self.max + offset)
    }

    /// Returns the rectangle grown by `dx` on each of the left and right
    /// edges and `dy` on each of the top and bottom, keeping the centre
    /// fixed.
    ///
    /// The extent therefore changes by `2 * dx` and `2 * dy`, not by `dx` and
    /// `dy`. Negative values shrink instead of grow — the usual way to inset
    /// a box by a margin.
    ///
    /// Shrinking past the centre returns None rather than an
    /// inverted or zero-extent rectangle.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0));
    ///
    /// let grown = r.inflate(2.0, 2.0).unwrap();
    /// assert_eq!(grown.min(), Point::new(-2.0, -2.0));
    /// assert_eq!(grown.size(), Size::new(14.0, 14.0)); // 10 + 2*2
    /// assert_eq!(grown.center(), r.center());
    ///
    /// // Negative insets.
    /// assert_eq!(r.inflate(-1.0, -1.0).unwrap().size(), Size::new(8.0, 8.0));
    ///
    /// // Over-shrinking collapses rather than inverting.
    /// assert!(r.inflate(-6.0, -6.0).is_none());
    /// ```
    pub fn inflate(self, dx: f64, dy: f64) -> Option<Rect> {
        let t = Vec2::new(dx, dy);
        let min = self.min - t;
        let max = self.max + t;
        if min.x >= max.x || min.y >= max.y {
            None
        } else {
            Some(Rect::from_points(min, max))
        }
    }

    /// Returns the rectangle repositioned so its [`origin`](Rect::origin) is
    /// `origin`, keeping the size.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.0));
    /// let moved = r.with_origin(Point::new(7.0, 7.0));
    ///
    /// assert_eq!(moved.origin(), Point::new(7.0, 7.0));
    /// assert_eq!(moved.size(), r.size());
    /// ```
    pub fn with_origin(self, origin: Point) -> Rect {
        Rect::from_origin_size(origin, self.size())
    }

    /// Returns the rectangle resized to `s`, keeping the
    /// [`origin`](Rect::origin) fixed.
    ///
    /// The origin corner stays put, so the rectangle grows away from it
    /// rather than around its centre — for the latter, build a new one with
    /// [`from_center_size`](Rect::from_center_size).
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect, Size};
    ///
    /// let r = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(4.0, 2.0));
    /// let resized = r.with_size(Size::new(10.0, 10.0));
    ///
    /// assert_eq!(resized.origin(), r.origin());
    /// assert_eq!(resized.max(), Point::new(11.0, 11.0));
    /// ```
    pub fn with_size(self, s: Size) -> Rect {
        Rect::from_origin_size(self.min, s)
    }

    /// Returns the smallest integer-aligned rectangle that contains `self`.
    ///
    /// Each `min` coordinate is rounded down and each `max` up, so the result
    /// always contains the original. This is the correct rounding for a
    /// damage or repaint region: covering slightly too much is harmless,
    /// covering too little leaves artefacts.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let r = Rect::from_points(Point::new(0.2, 0.7), Point::new(3.8, 3.1));
    /// assert_eq!(
    ///     r.round_out(),
    ///     Rect::from_points(Point::new(0.0, 0.0), Point::new(4.0, 4.0))
    /// );
    /// assert!(r.round_out().contains_rect(r));
    /// ```
    pub fn round_out(self) -> Rect {
        Rect::from_points(
            Point::new(self.x0().floor(), self.y0().floor()),
            Point::new(self.x1().ceil(), self.y1().ceil()),
        )
    }

    /// Returns the largest integer-aligned rectangle contained in `self`.
    ///
    /// Each `min` coordinate is rounded up and each `max` down — the
    /// conservative counterpart to [`round_out`](Rect::round_out), for when
    /// you must not exceed the original region.
    ///
    /// A rectangle narrower than the gap between two integers has no
    /// integer-aligned interior, so the result is None.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let r = Rect::from_points(Point::new(0.2, 0.7), Point::new(3.8, 3.1));
    /// assert_eq!(
    ///     r.round_in().unwrap(),
    ///     Rect::from_points(Point::new(1.0, 1.0), Point::new(3.0, 3.0))
    /// );
    /// assert!(r.contains_rect(r.round_in().unwrap()));
    ///
    /// let r = Rect::from_points(Point::new(0.2, 0.2), Point::new(0.8, 0.8));
    /// assert!(r.round_in().is_none());
    /// ```
    pub fn round_in(self) -> Option<Rect> {
        let min_x = self.x0().ceil();
        let max_x = self.x1().floor();
        let min_y = self.y0().ceil();
        let max_y = self.y1().floor();

        let min = Point::new(min_x, min_y);
        let max = Point::new(max_x, max_y);

        if min.x >= max.x || min.y >= max.y {
            None
        } else {
            Some(Rect::from_points(min, max))
        }
    }

    /// Returns the rectangle with every coordinate rounded to the nearest
    /// integer.
    ///
    /// Unlike [`round_out`](Rect::round_out) and
    /// [`round_in`](Rect::round_in), this makes no containment promise: the
    /// result may be slightly larger or smaller than the original on any
    /// edge. It can also collapse a thin rectangle to an empty one, when both
    /// edges of an axis round to the same integer — and unlike
    /// [`round_in`](Rect::round_in) it reports that in the value rather than
    /// as `None`, because rounding to nearest always has an answer.
    ///
    /// ```
    /// use grapsimo_geom::{Point, Rect};
    ///
    /// let r = Rect::from_points(Point::new(0.2, 0.7), Point::new(3.8, 3.1));
    /// assert_eq!(
    ///     r.round(),
    ///     Rect::from_points(Point::new(0.0, 1.0), Point::new(4.0, 3.0))
    /// );
    ///
    /// // A sliver can vanish.
    /// assert!(
    ///     Rect::from_points(Point::new(1.1, 0.0), Point::new(1.2, 4.0))
    ///         .round()
    ///         .is_empty()
    /// );
    /// ```
    pub fn round(self) -> Rect {
        let min_x = self.x0().round();
        let max_x = self.x1().round();
        let min_y = self.y0().round();
        let max_y = self.y1().round();

        let min = Point::new(min_x, min_y);
        let max = Point::new(max_x, max_y);

        Rect::from_points(min, max)
    }

    /// Returns the axis-aligned bounding box of the transformed rectangle.
    ///
    /// The four corners are transformed and a new axis-aligned box is fitted
    /// around them. For a translation or an axis-aligned scale that is exact,
    /// but for a rotation or a skew the true image is no longer axis-aligned,
    /// so the result is a *conservative over-estimate* — it contains the
    /// transformed shape but is larger than it.
    ///
    /// Because of that, the operation does not compose: transforming bounds
    /// twice grows the box more than transforming once by the composed
    /// matrix. Prefer applying the whole [`Affine`] chain first and taking
    /// bounds at the end.
    ///
    /// ```
    /// use grapsimo_geom::{Affine, ApproxEq, Point, Rect, Size, Vec2};
    /// use std::f64::consts::PI;
    ///
    /// let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.0));
    ///
    /// // Axis-aligned transforms are exact.
    /// let moved = r.transform_bounds(Affine::translate(Vec2::new(10.0, 0.0)));
    /// assert!(moved.approx_eq(r.translate(Vec2::new(10.0, 0.0))));
    /// assert!(r.transform_bounds(Affine::scale(2.0)).size().width.approx_eq(8.0));
    ///
    /// // A 45° rotation of a unit square gives a box sqrt(2) times as wide.
    /// let unit = Rect::from_center_size(Point::ORIGIN, Size::new(1.0, 1.0));
    /// let rotated = unit.transform_bounds(Affine::rotate(PI / 4.0));
    /// assert!(rotated.width().approx_eq(2.0f64.sqrt()));
    /// assert!(rotated.center().approx_eq(Point::ORIGIN));
    ///
    /// // ...and rotating a further 45° does not return the original box.
    /// assert!(rotated.transform_bounds(Affine::rotate(PI / 4.0)).width() > unit.width());
    /// ```
    pub fn transform_bounds(self, m: Affine) -> Rect {
        let c = self.corners();
        let c0 = m.transform_point(c[0]);
        let c1 = m.transform_point(c[1]);
        let c2 = m.transform_point(c[2]);
        let c3 = m.transform_point(c[3]);

        let x_min = c0.x.min(c1.x.min(c2.x.min(c3.x)));
        let x_max = c0.x.max(c1.x.max(c2.x.max(c3.x)));
        let y_min = c0.y.min(c1.y.min(c2.y.min(c3.y)));
        let y_max = c0.y.max(c1.y.max(c2.y.max(c3.y)));

        Self::from_points(Point::new(x_min, y_min), Point::new(x_max, y_max))
    }
}

/// Compares both corners against the epsilon.
///
/// Use this rather than `==` for rectangles that came out of a transform.
///
/// ```
/// use grapsimo_geom::{Affine, ApproxEq, Point, Rect, Size};
/// use std::f64::consts::PI;
///
/// let r = Rect::from_center_size(Point::ORIGIN, Size::new(2.0, 2.0));
/// let round_tripped = r
///     .transform_bounds(Affine::rotate(PI / 2.0))
///     .transform_bounds(Affine::rotate(-PI / 2.0));
///
/// assert!(round_tripped.approx_eq(r));
/// ```
impl ApproxEq for Rect {
    fn approx_eq_eps(self, other: Self, epsilon: f64) -> bool {
        (self.min.approx_eq_eps(other.min, epsilon)) && (self.max.approx_eq_eps(other.max, epsilon))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::test_util::assert_approx_eq;
    use std::f64::consts::PI;

    /// A 10x10 rect with its origin at the origin, used as the baseline in
    /// several tests below.
    fn unit_square_10() -> Rect {
        Rect::from_origin_size(Point::ORIGIN, Size::new(10.0, 10.0))
    }

    #[test]
    fn constructor_acessor() {
        let o = Point::ORIGIN;
        let p1 = Point::new(1.0, 1.0);
        let p2 = Point::new(2.0, 2.0);

        let r1 = Rect::from_points(p2, o);
        assert_eq!(r1.min(), o);
        assert_eq!(r1.max(), p2);
        assert_eq!(r1.center(), p1);
        assert_eq!(r1.width(), 2.0);
        assert_eq!(r1.height(), 2.0);

        let r2 = Rect::from_center_size(p1, Size::new(2.0, 2.0));
        assert_eq!(r1, r2);

        let p1 = Point::new(-28.0, 14.0);
        let p2 = Point::new(28.0, -14.0);

        let r1 = Rect::from_points(p1, p2);
        assert_eq!(r1.center(), o);
        assert_eq!(r1.min(), Point::new(-28.0, -14.0));
        assert_eq!(r1.max(), Point::new(28.0, 14.0));
        assert_eq!(r1.width(), 56.0);
        assert_eq!(r1.height(), 28.0);

        let r2 = Rect::from_origin_size(Point::new(-28.0, -14.0), Size::new(56.0, 28.0));
        assert_eq!(r1, r2);

        let r = Rect::from_points(Point::new(10.0, 10.0), Point::new(0.0, 0.0));
        assert_eq!(r.center(), Point::new(5.0, 5.0));
    }

    #[test]
    fn construction_normalizes() {
        let a = Point::new(4.0, 1.0);
        let b = Point::new(0.0, 3.0);

        // from_points is order independent.
        assert_eq!(Rect::from_points(a, b), Rect::from_points(b, a));
        assert_eq!(Rect::from_points(a, b).min(), Point::new(0.0, 1.0));
        assert_eq!(Rect::from_points(a, b).max(), Point::new(4.0, 3.0));

        // A negative extent grows backwards from the origin corner, and the
        // result is still normalized.
        let neg = Rect::from_origin_size(Point::new(10.0, 10.0), Size::new(-5.0, -5.0));
        assert_eq!(neg.min(), Point::new(5.0, 5.0));
        assert_eq!(neg.max(), Point::new(10.0, 10.0));
        assert_eq!(
            neg,
            Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(5.0, 5.0))
        );

        // from_center_size takes the magnitude, so the sign is irrelevant.
        let c = Point::new(1.0, 1.0);
        assert_eq!(
            Rect::from_center_size(c, Size::new(2.0, 2.0)),
            Rect::from_center_size(c, Size::new(-2.0, -2.0))
        );

        // Width and height are never negative, whatever the inputs.
        for r in [Rect::from_points(a, b), neg] {
            assert!(r.width() >= 0.0);
            assert!(r.height() >= 0.0);
        }
    }

    #[test]
    fn accessors() {
        let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));

        assert_eq!(r.x0(), 0.0);
        assert_eq!(r.x1(), 4.0);
        assert_eq!(r.y0(), 1.0);
        assert_eq!(r.y1(), 3.0);
        assert_eq!(r.origin(), r.min());
        assert_eq!(r.size(), Size::new(4.0, 2.0));
        assert_eq!(r.center(), Point::new(2.0, 2.0));
        assert_eq!(r.area(), 8.0);

        // origin + size round-trips back to the same rect.
        assert_eq!(Rect::from_origin_size(r.origin(), r.size()), r);

        // corners are in winding order starting at min; consecutive entries
        // share an edge.
        assert_eq!(
            r.corners(),
            [
                Point::new(0.0, 1.0),
                Point::new(0.0, 3.0),
                Point::new(4.0, 3.0),
                Point::new(4.0, 1.0),
            ]
        );

        // with_origin keeps the size, with_size keeps the origin.
        let moved = r.with_origin(Point::new(7.0, 7.0));
        assert_eq!(moved.origin(), Point::new(7.0, 7.0));
        assert_eq!(moved.size(), r.size());

        let resized = r.with_size(Size::new(10.0, 10.0));
        assert_eq!(resized.origin(), r.origin());
        assert_eq!(resized.size(), Size::new(10.0, 10.0));
    }

    #[test]
    fn contains() {
        let r = Rect::from_origin_size(Point::ORIGIN, Size::new(2.0, 2.0));

        // Interior.
        assert!(r.contains(Point::new(1.0, 1.0)));

        // Half-open: the min edges are inclusive...
        assert!(r.contains(Point::new(0.0, 0.0)));
        assert!(r.contains(Point::new(0.0, 1.0)));
        assert!(r.contains(Point::new(1.0, 0.0)));

        // ...and the max edges are exclusive.
        assert!(!r.contains(Point::new(2.0, 2.0)));
        assert!(!r.contains(Point::new(2.0, 1.0)));
        assert!(!r.contains(Point::new(1.0, 2.0)));

        // Fully outside, on each side.
        assert!(!r.contains(Point::new(-0.5, 1.0)));
        assert!(!r.contains(Point::new(2.5, 1.0)));
        assert!(!r.contains(Point::new(1.0, -0.5)));
        assert!(!r.contains(Point::new(1.0, 2.5)));

        // Half-open bounds mean abutting rects tile without overlap: the
        // shared edge belongs to exactly one of them.
        let left = Rect::from_points(Point::ORIGIN, Point::new(1.0, 1.0));
        let right = Rect::from_points(Point::new(1.0, 0.0), Point::new(2.0, 1.0));
        let seam = Point::new(1.0, 0.5);
        assert!(!left.contains(seam));
        assert!(right.contains(seam));
    }

    #[test]
    fn contains_rect_is_closed() {
        let outer = unit_square_10();
        let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));
        let flush = Rect::from_origin_size(Point::new(8.0, 8.0), Size::new(2.0, 2.0));
        let overlapping = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));

        assert!(outer.contains_rect(inner));
        assert!(!inner.contains_rect(outer));

        // Unlike `contains`, this comparison is closed on both ends: a rect
        // flush against the max edge still counts, and a rect contains itself.
        assert!(outer.contains_rect(flush));
        assert!(outer.contains_rect(outer));

        assert!(!outer.contains_rect(overlapping));
    }

    #[test]
    fn intersect() {
        let origin = Point::new(0.0, 0.0);
        let p1 = Point::new(5.0, 5.0);
        let p2 = Point::new(10.0, 10.0);
        let p3 = Point::new(-5.0, -5.0);
        let p4 = Point::new(-4.0, 3.0);
        let r1 = Rect::from_points(origin, p1);
        let r2 = Rect::from_points(origin, p2);
        let r3 = Rect::from_points(p3, origin);
        let r4 = Rect::from_points(p3, p4);

        assert_eq!(r2.intersect(r1).unwrap(), r1);
        assert!(r1.intersect(r3).is_none());
        assert!(r4.intersect(r1).is_none());

        // Partial overlap.
        let a = unit_square_10();
        let b = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
        assert_eq!(a.intersect(b).unwrap(), Rect::from_points(p1, p2));
        assert_eq!(a.intersect(b).unwrap().area(), 25.0);

        // Commutative and idempotent.
        assert_eq!(a.intersect(b), b.intersect(a));
        assert_eq!(a.intersect(a).unwrap(), a);

        // Containment yields the smaller rect.
        let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));
        assert_eq!(a.intersect(inner).unwrap(), inner);

        // Disjoint and edge-touching both return none
        let far = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
        let touching = Rect::from_origin_size(Point::new(10.0, 0.0), Size::new(5.0, 5.0));
        assert!(a.intersect(far).is_none());
        assert!(a.intersect(touching).is_none());
    }

    #[test]
    fn intersects_matches_intersect() {
        let a = unit_square_10();
        let overlapping = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
        let touching = Rect::from_origin_size(Point::new(10.0, 0.0), Size::new(5.0, 5.0));
        let corner_touching = Rect::from_origin_size(Point::new(10.0, 10.0), Size::new(5.0, 5.0));
        let disjoint = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));
        let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));

        // Touching along an edge or at a corner is not an intersection: the
        // shared region has zero area.
        for other in [overlapping, touching, corner_touching, disjoint, inner, a] {
            assert_eq!(
                a.intersects(other),
                a.intersect(other).is_some(),
                "intersects disagrees with intersect for {other:?}"
            );
            assert_eq!(a.intersects(other), other.intersects(a));
        }

        assert!(a.intersects(overlapping));
        assert!(a.intersects(inner));
        assert!(!a.intersects(touching));
        assert!(!a.intersects(corner_touching));
        assert!(!a.intersects(disjoint));
    }

    #[test]
    fn union() {
        let a = unit_square_10();
        let b = Rect::from_origin_size(Point::new(5.0, 5.0), Size::new(10.0, 10.0));
        let inner = Rect::from_origin_size(Point::new(1.0, 1.0), Size::new(2.0, 2.0));
        let far = Rect::from_origin_size(Point::new(20.0, 20.0), Size::new(1.0, 1.0));

        // Overlapping.
        assert_eq!(
            a.union(b),
            Rect::from_points(Point::ORIGIN, Point::new(15.0, 15.0))
        );

        // Disjoint: this is a bounding box, so the gap is covered too.
        assert_eq!(
            a.union(far),
            Rect::from_points(Point::ORIGIN, Point::new(21.0, 21.0))
        );

        // Containment absorbs the smaller rect.
        assert_eq!(a.union(inner), a);

        // Commutative and idempotent.
        assert_eq!(a.union(b), b.union(a));
        assert_eq!(a.union(a), a);

        // The union always contains both operands.
        for other in [b, inner, far] {
            let u = a.union(other);
            assert!(u.contains_rect(a));
            assert!(u.contains_rect(other));
        }
    }

    #[test]
    fn union_point() {
        let r = Rect::from_points(Point::new(3.0, 4.0), Point::new(4.0, 5.0));

        // A point already inside changes nothing.
        assert_eq!(r.union_point(Point::new(3.5, 4.5)), r);

        // A point outside grows the rect to reach it.
        assert_eq!(
            r.union_point(Point::new(-1.0, 9.0)),
            Rect::from_points(Point::new(-1.0, 4.0), Point::new(4.0, 9.0))
        );

        // Accumulating over several points from a non-empty seed.
        let bounds = [Point::new(-1.0, 9.0), Point::new(5.0, 0.0)]
            .into_iter()
            .fold(r, Rect::union_point);
        assert_eq!(bounds.min(), Point::new(-1.0, 0.0));
        assert_eq!(bounds.max(), Point::new(5.0, 9.0));

        // The growing point lands on the max edge, which `contains` excludes.
        // That is the half-open convention, not an accident.
        let grown = r.union_point(Point::new(9.0, 9.0));
        assert_eq!(grown.max(), Point::new(9.0, 9.0));
        assert!(!grown.contains(Point::new(9.0, 9.0)));
    }

    #[test]
    fn translate() {
        let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.0));
        let offset = Vec2::new(10.0, -5.0);
        let moved = r.translate(offset);

        assert_eq!(moved.origin(), Point::new(10.0, -5.0));
        assert_eq!(moved.size(), r.size());
        assert_eq!(moved.center(), r.center() + offset);

        // Invertible, and the zero vector is a no-op.
        assert_eq!(moved.translate(-offset), r);
        assert_eq!(r.translate(Vec2::ZERO), r);
    }

    #[test]
    fn inflate() {
        let r = unit_square_10();

        // Positive margin grows: dx is applied to *each* side, so the extent
        // changes by 2 * dx.
        let grown = r.inflate(2.0, 2.0).unwrap();
        assert_eq!(grown.min(), Point::new(-2.0, -2.0));
        assert_eq!(grown.max(), Point::new(12.0, 12.0));
        assert_eq!(grown.size(), Size::new(14.0, 14.0));

        // The center is fixed, and growing then shrinking round-trips.
        assert_eq!(grown.center(), r.center());
        assert_eq!(grown.inflate(-2.0, -2.0).unwrap(), r);

        // Zero is a no-op.
        assert_eq!(r.inflate(0.0, 0.0).unwrap(), r);

        // Negative margin shrinks.
        let shrunk = r.inflate(-1.0, -1.0).unwrap();
        assert_eq!(
            shrunk,
            Rect::from_points(Point::new(1.0, 1.0), Point::new(9.0, 9.0))
        );
        assert_eq!(shrunk.center(), r.center());
        assert!(r.contains_rect(shrunk));

        // The axes are independent.
        let stretched = r.inflate(2.0, 0.0).unwrap();
        assert_eq!(stretched.size(), Size::new(14.0, 10.0));

        // Shrinking exactly to the center collapses, rather than producing a
        // zero-extent rect...
        assert!(r.inflate(-5.0, -5.0).is_none());

        // ...and over-shrinking collapses rather than inverting.
        assert!(r.inflate(-6.0, -6.0).is_none());
        assert!(r.inflate(-20.0, -1.0).is_none());

        // Collapse on either axis alone is enough.
        assert!(r.inflate(-1.0, -6.0).is_none());
    }

    #[test]
    fn rounding() {
        let r = Rect::from_points(Point::new(0.2, 0.7), Point::new(3.8, 3.1));

        // round_out grows to the enclosing integer box.
        assert_eq!(
            r.round_out(),
            Rect::from_points(Point::new(0.0, 0.0), Point::new(4.0, 4.0))
        );
        assert!(r.round_out().contains_rect(r));

        // round_in shrinks to the enclosed integer box.
        assert_eq!(
            r.round_in().unwrap(),
            Rect::from_points(Point::new(1.0, 1.0), Point::new(3.0, 3.0))
        );
        assert!(r.contains_rect(r.round_in().unwrap()));

        // round is nearest, and promises neither containment direction.
        assert_eq!(
            r.round(),
            Rect::from_points(Point::new(0.0, 1.0), Point::new(4.0, 3.0))
        );

        // round can collapse a sliver whose edges round to the same integer.
        assert!(
            Rect::from_points(Point::new(1.1, 0.0), Point::new(1.2, 4.0))
                .round()
                .is_empty()
        );

        // An already-integral rect is a fixed point of all three.
        let integral = Rect::from_points(Point::new(-2.0, 1.0), Point::new(5.0, 8.0));
        assert_eq!(integral.round_out(), integral);
        assert_eq!(integral.round_in().unwrap(), integral);
        assert_eq!(integral.round(), integral);

        // Negative coordinates round away from zero on the low side.
        let neg = Rect::from_points(Point::new(-1.2, -1.8), Point::new(1.0, 1.0));
        assert_eq!(neg.round_out().min(), Point::new(-2.0, -2.0));
        assert_eq!(neg.round_in().unwrap().min(), Point::new(-1.0, -1.0));
    }

    #[test]
    fn transform() {
        let r = Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 2.0));

        // Identity leaves the bounds alone.
        assert_approx_eq!(r.transform_bounds(Affine::IDENTITY), r);

        // Translation is exact, and agrees with `translate`.
        let offset = Vec2::new(10.0, -5.0);
        assert_approx_eq!(
            r.transform_bounds(Affine::translate(offset)),
            r.translate(offset)
        );

        // Axis-aligned scaling is exact.
        assert_approx_eq!(
            r.transform_bounds(Affine::scale(2.0)),
            Rect::from_origin_size(Point::ORIGIN, Size::new(8.0, 4.0))
        );
        assert_approx_eq!(
            r.transform_bounds(Affine::scale_xy(1.0, 3.0)),
            Rect::from_origin_size(Point::ORIGIN, Size::new(4.0, 6.0))
        );

        // A quarter turn is also exact: it maps the box onto itself with the
        // extents swapped.
        let quarter = r.transform_bounds(Affine::rotate(PI / 2.0));
        assert_approx_eq!(quarter.width(), 2.0);
        assert_approx_eq!(quarter.height(), 4.0);
        assert_approx_eq!(quarter.area(), r.area());

        // A 45 degree rotation is where the bounds become lossy: the true
        // image is no longer axis-aligned, so the box that fits around it is
        // sqrt(2) times wider than the original square.
        let unit = Rect::from_center_size(Point::ORIGIN, Size::new(1.0, 1.0));
        let diagonal = unit.transform_bounds(Affine::rotate(PI / 4.0));
        assert_approx_eq!(diagonal.width(), 2.0f64.sqrt());
        assert_approx_eq!(diagonal.height(), 2.0f64.sqrt());
        assert_approx_eq!(diagonal.center(), Point::ORIGIN);

        // Rotation about the center never moves the center.
        let c = r.center();
        assert_approx_eq!(r.transform_bounds(Affine::rotate_about(c, 0.3)).center(), c);

        // The bounds always contain the transformed corners, whatever the
        // transform.
        let m = Affine::rotate(0.7).then(Affine::scale_xy(2.0, 0.5));
        let bounds = r.transform_bounds(m);
        for corner in r.corners() {
            let p = m.transform_point(corner);
            assert!(
                bounds.contains_rect(Rect::from_points(p, p)),
                "transformed corner {p:?} escaped the bounds {bounds:?}"
            );
        }

        // Because the result is a conservative over-estimate, the operation
        // does not compose: taking bounds twice grows the box more than
        // transforming once by the composed matrix.
        let twice = diagonal.transform_bounds(Affine::rotate(PI / 4.0));
        let once = unit.transform_bounds(Affine::rotate(PI / 2.0));
        assert!(twice.width() > once.width());
        assert_approx_eq!(once.width(), 1.0);
        assert_approx_eq!(twice.width(), 2.0);
    }

    #[test]
    fn empty() {
        let r = unit_square_10();
        let degenerate = Rect::from_points(Point::new(9.0, 9.0), Point::new(9.0, 9.0));
        let zero_width = Rect::from_points(Point::new(0.0, 0.0), Point::new(0.0, 5.0));

        // Zero extent on either axis is empty, wherever the rect sits.
        assert!(degenerate.is_empty());
        assert!(zero_width.is_empty());
        assert!(!r.is_empty());

        assert_eq!(degenerate.area(), 0.0);
        assert_eq!(zero_width.area(), 0.0);

        assert!(!degenerate.contains(Point::new(9.0, 9.0)));
        assert!(!zero_width.contains(Point::new(0.0, 2.0)));

        // Intersecting with it always yields none.
        for e in [degenerate, zero_width] {
            assert!(r.intersect(e).is_none());
            assert!(e.intersect(r).is_none());
        }

        // `intersects` agrees for these two, but see
        // `intersects_agrees_with_intersect_for_degenerate` below.
        assert!(!r.intersects(zero_width));

        // Union treats an empty operand as the identity and skips it, so an
        // empty rect's position never leaks into the result.
        for e in [degenerate, zero_width] {
            assert_eq!(r.union(e), r);
            assert_eq!(e.union(r), r);
        }

        // `contains_rect` is closed, so an empty rect positioned inside
        // another rect still counts as contained. This follows from the
        // definition rather than being independently useful; it is pinned
        // here so a change to the comparison shows up as a test failure.
        assert!(r.contains_rect(degenerate));
        assert!(!r.contains_rect(Rect::from_points(
            Point::new(99.0, 99.0),
            Point::new(99.0, 99.0)
        )));

        // Inflating an empty rect can revive it, since the center is still
        // well defined.
        assert_eq!(
            degenerate.inflate(1.0, 1.0).unwrap(),
            Rect::from_points(Point::new(8.0, 8.0), Point::new(10.0, 10.0))
        );

        // Transforming an empty rect keeps it empty.
        assert!(degenerate.transform_bounds(Affine::scale(3.0)).is_empty());
    }

    #[test]
    fn approx_eq() {
        let r = Rect::from_points(Point::new(0.1, -4.2), Point::new(3.0, 5.0));
        let nudged = Rect::from_points(Point::new(0.1 + 9.9e-11, -4.2), Point::new(3.0, 5.0));

        assert!(r != nudged);
        assert!(r.approx_eq(nudged));
        assert!(!r.approx_eq_eps(nudged, 1e-11));

        // A rotation round-trip only lands back on the original approximately.
        let round_tripped = r
            .transform_bounds(Affine::rotate(PI / 2.0))
            .transform_bounds(Affine::rotate(-PI / 2.0));
        assert!(round_tripped.approx_eq(r));
    }

    #[test]
    fn bounding() {
        // A single point gives a degenerate rect at that point. It has a
        // position but no area, so it is `is_empty()`.
        let p = Point::new(3.0, 4.0);
        let single = Rect::bounding([p]).unwrap();
        assert_eq!(single.min(), p);
        assert_eq!(single.max(), p);
        assert!(single.is_empty());

        let points = [
            Point::new(13.0, 14.0),
            Point::new(9.0, 19.0),
            Point::new(15.0, 10.0),
        ];
        let bounds = Rect::bounding(points).unwrap();
        assert_eq!(bounds.min(), Point::new(9.0, 10.0));
        assert_eq!(bounds.max(), Point::new(15.0, 19.0));

        // Every input point is enclosed. `contains` would reject the ones on
        // the max edges, so compare against a degenerate rect instead, which
        // `contains_rect` treats as closed.
        for p in points {
            assert!(
                bounds.contains_rect(Rect::from_points(p, p)),
                "{p:?} escaped the bounds {bounds:?}"
            );
        }

        // ...and the result is tight: shrinking it on any side would drop a
        // point, so every edge is touched by at least one input.
        assert!(points.iter().any(|p| p.x == bounds.x0()));
        assert!(points.iter().any(|p| p.x == bounds.x1()));
        assert!(points.iter().any(|p| p.y == bounds.y0()));
        assert!(points.iter().any(|p| p.y == bounds.y1()));

        // Order independent, and repeated points make no difference.
        let mut reversed = points;
        reversed.reverse();
        assert_eq!(Rect::bounding(reversed).unwrap(), bounds);
        assert_eq!(Rect::bounding([p, p, p]), Rect::bounding([p]));

        // Negative coordinates and points straddling the origin.
        assert_eq!(
            Rect::bounding([Point::new(-4.0, -2.0), Point::new(1.0, 3.0)]).unwrap(),
            Rect::from_points(Point::new(-4.0, -2.0), Point::new(1.0, 3.0))
        );

        // Bounding a rect's own corners reproduces the rect.
        let r = Rect::from_points(Point::new(4.0, 1.0), Point::new(0.0, 3.0));
        assert_eq!(Rect::bounding(r.corners()).unwrap(), r);

        // Accepts anything iterable over points: arrays, Vecs, and adaptor
        // chains alike.
        assert_eq!(Rect::bounding(points.to_vec()).unwrap(), bounds);
        assert_eq!(Rect::bounding(points.iter().copied()).unwrap(), bounds);
        assert_eq!(
            Rect::bounding(points.into_iter().filter(|p| p.x > 10.0)).unwrap(),
            Rect::from_points(Point::new(13.0, 10.0), Point::new(15.0, 14.0))
        );
    }

    #[test]
    fn intersects_agrees_with_intersect_for_degenerate() {
        let r = unit_square_10();
        let degenerate = Rect::from_points(Point::new(9.0, 9.0), Point::new(9.0, 9.0));

        assert!(degenerate.is_empty());

        assert!(!r.intersects(degenerate));
        assert!(!degenerate.intersects(r));
    }

    #[test]
    fn round_in_collapses_sub_integer_rect() {
        let sliver = Rect::from_points(Point::new(0.2, 0.2), Point::new(0.8, 0.8));

        assert!(sliver.round_in().is_none());
    }
}
