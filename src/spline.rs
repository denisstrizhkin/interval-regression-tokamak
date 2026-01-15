//! Spline interpolation module for temperature profile analysis.
//!
//! Implements piecewise linear (1st order) and cubic (3rd order) spline
//! interpolation to describe temperature profiles T_before and T_after
//! sawtooth oscillations. Supports interval-valued data for creating
//! "compatibility corridors".

use crate::interval::Interval;

/// A point in the temperature profile.
#[derive(Debug, Clone, Copy)]
pub struct DataPoint {
    pub radius: f64,      // Radial position (mm or normalized)
    pub temperature: f64, // Temperature value (normalized T_e/<Te>)
}

/// An interval-valued data point for uncertainty analysis.
#[derive(Debug, Clone, Copy)]
pub struct IntervalDataPoint {
    pub radius: f64,
    pub temperature: Interval,
}

impl IntervalDataPoint {
    pub fn new(radius: f64, temp_lower: f64, temp_upper: f64) -> Self {
        Self {
            radius,
            temperature: Interval::new(temp_lower, temp_upper),
        }
    }

    pub fn from_point_and_error(radius: f64, temp: f64, error: f64) -> Self {
        Self {
            radius,
            temperature: Interval::from_center_radius(temp, error),
        }
    }
}

/// Piecewise linear interpolation (1st order spline).
#[derive(Debug, Clone)]
pub struct LinearSpline {
    points: Vec<DataPoint>,
}

impl LinearSpline {
    /// Creates a new linear spline from sorted data points.
    /// Points must be sorted by radius in ascending order.
    pub fn new(mut points: Vec<DataPoint>) -> Self {
        points.sort_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap());
        // Remove duplicate radii by keeping the last occurrence
        points.dedup_by(|a, b| (a.radius - b.radius).abs() < 1e-10);
        Self { points }
    }

    /// Interpolates the temperature at a given radius.
    /// Returns None if the radius is outside the data range.
    pub fn interpolate(&self, radius: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        if self.points.len() == 1 {
            return Some(self.points[0].temperature);
        }

        // Find the interval containing the radius
        for i in 0..self.points.len() - 1 {
            let p0 = &self.points[i];
            let p1 = &self.points[i + 1];

            if radius >= p0.radius && radius <= p1.radius {
                let t = (radius - p0.radius) / (p1.radius - p0.radius);
                return Some(p0.temperature + t * (p1.temperature - p0.temperature));
            }
        }

        // Extrapolation at boundaries
        if radius < self.points[0].radius {
            // Linear extrapolation from first two points
            if self.points.len() >= 2 {
                let p0 = &self.points[0];
                let p1 = &self.points[1];
                let slope = (p1.temperature - p0.temperature) / (p1.radius - p0.radius);
                Some(p0.temperature + slope * (radius - p0.radius))
            } else {
                Some(self.points[0].temperature)
            }
        } else {
            // Linear extrapolation from last two points
            let n = self.points.len();
            if n >= 2 {
                let p0 = &self.points[n - 2];
                let p1 = &self.points[n - 1];
                let slope = (p1.temperature - p0.temperature) / (p1.radius - p0.radius);
                Some(p1.temperature + slope * (radius - p1.radius))
            } else {
                Some(self.points[n - 1].temperature)
            }
        }
    }

    /// Returns the radial range of the spline.
    pub fn range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            None
        } else {
            Some((self.points[0].radius, self.points.last().unwrap().radius))
        }
    }
}

/// Natural cubic spline interpolation (3rd order spline).
#[derive(Debug, Clone)]
pub struct CubicSpline {
    points: Vec<DataPoint>,
    // Second derivatives at each point (for natural spline)
    second_derivs: Vec<f64>,
}

impl CubicSpline {
    /// Creates a new natural cubic spline from sorted data points.
    pub fn new(mut points: Vec<DataPoint>) -> Self {
        points.sort_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap());
        points.dedup_by(|a, b| (a.radius - b.radius).abs() < 1e-10);

        let second_derivs = Self::compute_second_derivatives(&points);

        Self {
            points,
            second_derivs,
        }
    }

    /// Computes the second derivatives using the tridiagonal algorithm.
    fn compute_second_derivatives(points: &[DataPoint]) -> Vec<f64> {
        let n = points.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n - 1];

        // Natural spline boundary condition: y2[0] = 0
        y2[0] = 0.0;
        u[0] = 0.0;

        // Forward pass
        for i in 1..n - 1 {
            let h_prev = points[i].radius - points[i - 1].radius;
            let h_next = points[i + 1].radius - points[i].radius;
            let sig = h_prev / (h_prev + h_next);

            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;

            let dy_prev = (points[i].temperature - points[i - 1].temperature) / h_prev;
            let dy_next = (points[i + 1].temperature - points[i].temperature) / h_next;
            u[i] = (6.0 * (dy_next - dy_prev) / (h_prev + h_next) - sig * u[i - 1]) / p;
        }

        // Natural spline boundary condition: y2[n-1] = 0
        y2[n - 1] = 0.0;

        // Back substitution
        for k in (0..n - 1).rev() {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }

        y2
    }

    /// Interpolates the temperature at a given radius.
    pub fn interpolate(&self, radius: f64) -> Option<f64> {
        if self.points.len() < 2 {
            return self.points.first().map(|p| p.temperature);
        }

        // Find the interval containing the radius
        let mut lo = 0;
        let mut hi = self.points.len() - 1;

        // Handle extrapolation
        if radius <= self.points[0].radius {
            lo = 0;
            hi = 1;
        } else if radius >= self.points[self.points.len() - 1].radius {
            lo = self.points.len() - 2;
            hi = self.points.len() - 1;
        } else {
            // Binary search
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if self.points[mid].radius > radius {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
        }

        let h = self.points[hi].radius - self.points[lo].radius;
        if h.abs() < 1e-15 {
            return Some(self.points[lo].temperature);
        }

        let a = (self.points[hi].radius - radius) / h;
        let b = (radius - self.points[lo].radius) / h;

        let result = a * self.points[lo].temperature
            + b * self.points[hi].temperature
            + ((a * a * a - a) * self.second_derivs[lo] + (b * b * b - b) * self.second_derivs[hi])
                * (h * h)
                / 6.0;

        Some(result)
    }

    /// Returns the radial range of the spline.
    pub fn range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            None
        } else {
            Some((self.points[0].radius, self.points.last().unwrap().radius))
        }
    }
}

/// Trait for spline interpolation (allows polymorphism over spline types).
pub trait Spline {
    fn interpolate(&self, radius: f64) -> Option<f64>;
    fn range(&self) -> Option<(f64, f64)>;
}

impl Spline for LinearSpline {
    fn interpolate(&self, radius: f64) -> Option<f64> {
        self.interpolate(radius)
    }

    fn range(&self) -> Option<(f64, f64)> {
        self.range()
    }
}

impl Spline for CubicSpline {
    fn interpolate(&self, radius: f64) -> Option<f64> {
        self.interpolate(radius)
    }

    fn range(&self) -> Option<(f64, f64)> {
        self.range()
    }
}

/// Interval-valued spline for compatibility corridors.
/// Uses upper and lower envelope splines to bound temperature uncertainty.
#[derive(Debug, Clone)]
pub struct IntervalSpline<S: Spline + Clone> {
    pub lower_spline: S,
    pub upper_spline: S,
}

impl<S: Spline + Clone> IntervalSpline<S> {
    /// Interpolates to get an interval at a given radius.
    pub fn interpolate(&self, radius: f64) -> Option<Interval> {
        let lower = self.lower_spline.interpolate(radius)?;
        let upper = self.upper_spline.interpolate(radius)?;

        // Ensure lower <= upper
        if lower <= upper {
            Some(Interval::new(lower, upper))
        } else {
            Some(Interval::new(upper, lower))
        }
    }

    /// Returns the combined radial range of both splines.
    pub fn range(&self) -> Option<(f64, f64)> {
        let r1 = self.lower_spline.range()?;
        let r2 = self.upper_spline.range()?;
        Some((r1.0.max(r2.0), r1.1.min(r2.1)))
    }
}

/// Creates upper and lower envelope data points from interval data.
pub fn create_envelope_points(
    interval_points: &[IntervalDataPoint],
) -> (Vec<DataPoint>, Vec<DataPoint>) {
    let lower_points: Vec<DataPoint> = interval_points
        .iter()
        .map(|p| DataPoint {
            radius: p.radius,
            temperature: p.temperature.lower,
        })
        .collect();

    let upper_points: Vec<DataPoint> = interval_points
        .iter()
        .map(|p| DataPoint {
            radius: p.radius,
            temperature: p.temperature.upper,
        })
        .collect();

    (lower_points, upper_points)
}

/// Creates a linear interval spline from interval data points.
pub fn create_linear_interval_spline(points: &[IntervalDataPoint]) -> IntervalSpline<LinearSpline> {
    let (lower_points, upper_points) = create_envelope_points(points);
    IntervalSpline {
        lower_spline: LinearSpline::new(lower_points),
        upper_spline: LinearSpline::new(upper_points),
    }
}

/// Creates a cubic interval spline from interval data points.
pub fn create_cubic_interval_spline(points: &[IntervalDataPoint]) -> IntervalSpline<CubicSpline> {
    let (lower_points, upper_points) = create_envelope_points(points);
    IntervalSpline {
        lower_spline: CubicSpline::new(lower_points),
        upper_spline: CubicSpline::new(upper_points),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let points = vec![
            DataPoint {
                radius: 0.0,
                temperature: 0.0,
            },
            DataPoint {
                radius: 1.0,
                temperature: 1.0,
            },
            DataPoint {
                radius: 2.0,
                temperature: 2.0,
            },
        ];
        let spline = LinearSpline::new(points);

        assert!((spline.interpolate(0.5).unwrap() - 0.5).abs() < 1e-10);
        assert!((spline.interpolate(1.5).unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_interpolation() {
        let points = vec![
            DataPoint {
                radius: 0.0,
                temperature: 0.0,
            },
            DataPoint {
                radius: 1.0,
                temperature: 1.0,
            },
            DataPoint {
                radius: 2.0,
                temperature: 4.0,
            },
            DataPoint {
                radius: 3.0,
                temperature: 9.0,
            },
        ];
        let spline = CubicSpline::new(points);

        // For a parabola-like curve, check that interpolation is smooth
        let mid = spline.interpolate(1.5).unwrap();
        assert!(mid > 1.0 && mid < 4.0);
    }
}
