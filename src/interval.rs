//! Interval arithmetic module for plasma physics analysis.
//!
//! Implements classical interval arithmetic (IR) with operations for
//! addition, subtraction, multiplication, and division, along with
//! set-theoretic operations like intersection, union, and Jaccard index.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A closed interval [lower, upper] representing uncertainty bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    /// Creates a new interval [lower, upper].
    /// Panics if lower > upper.
    pub fn new(lower: f64, upper: f64) -> Self {
        assert!(
            lower <= upper,
            "Invalid interval: lower ({}) > upper ({})",
            lower,
            upper
        );
        Self { lower, upper }
    }

    /// Creates a point interval [value, value].
    pub fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Creates an interval from a center and radius (half-width).
    pub fn from_center_radius(center: f64, radius: f64) -> Self {
        Self::new(center - radius.abs(), center + radius.abs())
    }

    /// Returns the width (diameter) of the interval.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Returns the midpoint (center) of the interval.
    pub fn midpoint(&self) -> f64 {
        0.5 * (self.lower + self.upper)
    }

    /// Returns the radius (half-width) of the interval.
    pub fn radius(&self) -> f64 {
        0.5 * self.width()
    }

    /// Checks if this interval contains a point.
    pub fn contains_point(&self, x: f64) -> bool {
        self.lower <= x && x <= self.upper
    }

    /// Checks if this interval contains another interval (inclusion relation).
    pub fn contains(&self, other: &Interval) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    /// Checks if two intervals overlap.
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }

    /// Returns the intersection of two intervals, or None if disjoint.
    pub fn intersection(&self, other: &Interval) -> Option<Interval> {
        if self.overlaps(other) {
            Some(Interval::new(
                self.lower.max(other.lower),
                self.upper.min(other.upper),
            ))
        } else {
            None
        }
    }

    /// Returns the hull (smallest interval containing both).
    pub fn hull(&self, other: &Interval) -> Interval {
        Interval::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }

    /// Computes the Jaccard index J_I = width(intersection) / width(union).
    /// Returns 0 if the intervals are disjoint.
    pub fn jaccard_index(&self, other: &Interval) -> f64 {
        if let Some(intersection) = self.intersection(other) {
            let union = self.hull(other);
            let union_width = union.width();
            if union_width == 0.0 {
                1.0 // Both are point intervals at the same location
            } else {
                intersection.width() / union_width
            }
        } else {
            0.0
        }
    }

    /// Computes the relative Jaccard measure J_IX = width(intersection) / width(self).
    /// Returns 0 if the intervals are disjoint.
    pub fn relative_jaccard(&self, other: &Interval) -> f64 {
        if let Some(intersection) = self.intersection(other) {
            let self_width = self.width();
            if self_width == 0.0 {
                if other.contains_point(self.lower) {
                    1.0
                } else {
                    0.0
                }
            } else {
                intersection.width() / self_width
            }
        } else {
            0.0
        }
    }

    /// Checks if the interval is valid (not NaN or infinite in unexpected ways).
    pub fn is_valid(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6}, {:.6}]", self.lower, self.upper)
    }
}

// Interval Arithmetic: Addition
impl Add for Interval {
    type Output = Interval;

    fn add(self, other: Interval) -> Interval {
        Interval::new(self.lower + other.lower, self.upper + other.upper)
    }
}

impl Add<f64> for Interval {
    type Output = Interval;

    fn add(self, scalar: f64) -> Interval {
        Interval::new(self.lower + scalar, self.upper + scalar)
    }
}

// Interval Arithmetic: Subtraction
impl Sub for Interval {
    type Output = Interval;

    fn sub(self, other: Interval) -> Interval {
        Interval::new(self.lower - other.upper, self.upper - other.lower)
    }
}

impl Sub<f64> for Interval {
    type Output = Interval;

    fn sub(self, scalar: f64) -> Interval {
        Interval::new(self.lower - scalar, self.upper - scalar)
    }
}

// Interval Arithmetic: Negation
impl Neg for Interval {
    type Output = Interval;

    fn neg(self) -> Interval {
        Interval::new(-self.upper, -self.lower)
    }
}

// Interval Arithmetic: Multiplication
impl Mul for Interval {
    type Output = Interval;

    fn mul(self, other: Interval) -> Interval {
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        let min = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Interval::new(min, max)
    }
}

impl Mul<f64> for Interval {
    type Output = Interval;

    fn mul(self, scalar: f64) -> Interval {
        if scalar >= 0.0 {
            Interval::new(self.lower * scalar, self.upper * scalar)
        } else {
            Interval::new(self.upper * scalar, self.lower * scalar)
        }
    }
}

// Interval Arithmetic: Division
impl Div for Interval {
    type Output = Option<Interval>;

    fn div(self, other: Interval) -> Option<Interval> {
        // Division is undefined if 0 is in the divisor interval
        if other.lower <= 0.0 && 0.0 <= other.upper {
            None
        } else {
            let quotients = [
                self.lower / other.lower,
                self.lower / other.upper,
                self.upper / other.lower,
                self.upper / other.upper,
            ];
            let min = quotients.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = quotients.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some(Interval::new(min, max))
        }
    }
}

impl Div<f64> for Interval {
    type Output = Option<Interval>;

    fn div(self, scalar: f64) -> Option<Interval> {
        if scalar == 0.0 {
            None
        } else if scalar > 0.0 {
            Some(Interval::new(self.lower / scalar, self.upper / scalar))
        } else {
            Some(Interval::new(self.upper / scalar, self.lower / scalar))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let i = Interval::new(1.0, 3.0);
        assert_eq!(i.lower, 1.0);
        assert_eq!(i.upper, 3.0);
    }

    #[test]
    fn test_width_and_midpoint() {
        let i = Interval::new(2.0, 6.0);
        assert_eq!(i.width(), 4.0);
        assert_eq!(i.midpoint(), 4.0);
        assert_eq!(i.radius(), 2.0);
    }

    #[test]
    fn test_addition() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let c = a + b;
        assert_eq!(c.lower, 4.0);
        assert_eq!(c.upper, 6.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(-1.0, 2.0);
        let c = a * b;
        assert_eq!(c.lower, -4.0);
        assert_eq!(c.upper, 6.0);
    }

    #[test]
    fn test_jaccard_index() {
        let a = Interval::new(0.0, 2.0);
        let b = Interval::new(1.0, 3.0);
        let ji = a.jaccard_index(&b);
        // Intersection is [1, 2], width = 1
        // Union hull is [0, 3], width = 3
        // Jaccard = 1/3
        assert!((ji - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_intersection() {
        let a = Interval::new(0.0, 2.0);
        let b = Interval::new(1.0, 3.0);
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.lower, 1.0);
        assert_eq!(inter.upper, 2.0);
    }
}
