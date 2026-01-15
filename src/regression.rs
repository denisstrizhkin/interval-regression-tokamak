//! Interval regression module.
//!
//! Implements interval-valued regression to model the dependency of
//! the inversion radius R_inv on the magnetic field ratio B_T/I_P,
//! enabling forecasting for future fusion devices.

use crate::interval::Interval;
use crate::inversion::InversionAnalysis;

/// Result of interval regression analysis.
#[derive(Debug, Clone)]
pub struct IntervalRegression {
    /// Slope of the central line (point estimate)
    pub slope: f64,
    /// Intercept of the central line
    pub intercept: f64,
    /// Interval for the slope [slope_lower, slope_upper]
    pub slope_interval: Interval,
    /// Interval for the intercept [intercept_lower, intercept_upper]
    pub intercept_interval: Interval,
    /// R-squared value for the point estimate fit
    pub r_squared: f64,
    /// Data points used in regression
    pub data_points: Vec<(f64, f64)>, // (x = B_T/I_P, y = R_inv)
    /// Interval bands at each data point
    pub interval_bands: Vec<(f64, Interval)>, // (x, [y_lower, y_upper])
}

impl IntervalRegression {
    /// Predicts R_inv for a given B_T/I_P ratio (point estimate).
    pub fn predict(&self, bt_ip: f64) -> f64 {
        self.slope * bt_ip + self.intercept
    }

    /// Predicts R_inv interval for a given B_T/I_P ratio.
    pub fn predict_interval(&self, bt_ip: f64) -> Interval {
        // Use interval arithmetic for prediction uncertainty
        let slope_contribution = self.slope_interval * Interval::point(bt_ip);
        let prediction = slope_contribution + self.intercept_interval;
        prediction
    }

    /// Generates forecast data for a range of B_T/I_P values.
    pub fn forecast(
        &self,
        bt_ip_min: f64,
        bt_ip_max: f64,
        num_points: usize,
    ) -> Vec<ForecastPoint> {
        let step = (bt_ip_max - bt_ip_min) / (num_points as f64 - 1.0);

        (0..num_points)
            .map(|i| {
                let bt_ip = bt_ip_min + i as f64 * step;
                let point_estimate = self.predict(bt_ip);
                let interval = self.predict_interval(bt_ip);
                ForecastPoint {
                    bt_ip,
                    r_inv_point: point_estimate,
                    r_inv_lower: interval.lower,
                    r_inv_upper: interval.upper,
                }
            })
            .collect()
    }
}

/// A single point in the forecast curve.
#[derive(Debug, Clone)]
pub struct ForecastPoint {
    pub bt_ip: f64,
    pub r_inv_point: f64,
    pub r_inv_lower: f64,
    pub r_inv_upper: f64,
}

/// Performs ordinary least squares regression on interval-valued data.
/// Uses the midpoints for the central estimate and bounds for uncertainty bands.
pub fn interval_least_squares(analyses: &[InversionAnalysis]) -> Option<IntervalRegression> {
    if analyses.len() < 2 {
        return None;
    }

    // Extract data points: x = B_T/I_P, y = R_inv_point
    let data_points: Vec<(f64, f64)> = analyses
        .iter()
        .map(|a| (a.bt_ip_ratio, a.r_inv_point))
        .collect();

    let n = data_points.len() as f64;

    // Compute means
    let x_mean = data_points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_mean = data_points.iter().map(|(_, y)| y).sum::<f64>() / n;

    // Compute sums for OLS
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for (x, y) in &data_points {
        let dx = x - x_mean;
        let dy = y - y_mean;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    if ss_xx == 0.0 {
        return None;
    }

    // OLS estimates
    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    // R-squared
    let r_squared = if ss_yy > 0.0 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        1.0
    };

    // Compute residuals and standard error
    let mut residual_ss = 0.0;
    for (x, y) in &data_points {
        let predicted = slope * x + intercept;
        let residual = y - predicted;
        residual_ss += residual * residual;
    }

    let dof = n - 2.0;
    let se = if dof > 0.0 {
        (residual_ss / dof).sqrt()
    } else {
        0.0
    };

    // Standard errors of coefficients
    let se_slope = se / ss_xx.sqrt();
    let se_intercept = se * (1.0 / n + x_mean * x_mean / ss_xx).sqrt();

    // 95% confidence interval (approx t-value = 2 for moderate sample sizes)
    let t_value = 2.0;
    let slope_interval = Interval::new(slope - t_value * se_slope, slope + t_value * se_slope);
    let intercept_interval = Interval::new(
        intercept - t_value * se_intercept,
        intercept + t_value * se_intercept,
    );

    // Compute interval bands at each data point
    let interval_bands: Vec<(f64, Interval)> = analyses
        .iter()
        .map(|a| {
            let x = a.bt_ip_ratio;
            let r_inv_interval = Interval::new(a.r_inv_low, a.r_inv_high);
            (x, r_inv_interval)
        })
        .collect();

    Some(IntervalRegression {
        slope,
        intercept,
        slope_interval,
        intercept_interval,
        r_squared,
        data_points,
        interval_bands,
    })
}

/// Computes robust interval regression using the min/max method.
/// This method determines the bounding lines that contain all interval data.
pub fn robust_interval_regression(analyses: &[InversionAnalysis]) -> Option<IntervalRegression> {
    if analyses.len() < 2 {
        return None;
    }

    // First compute the central OLS estimate
    let ols = interval_least_squares(analyses)?;

    // Extract interval bounds for each data point
    let interval_data: Vec<(f64, Interval)> = analyses
        .iter()
        .map(|a| {
            // Use low and high estimates
            (a.bt_ip_ratio, Interval::new(a.r_inv_low, a.r_inv_high))
        })
        .collect();

    // Compute min and max slopes from interval constraints
    let mut slope_lower = f64::NEG_INFINITY;
    let mut slope_upper = f64::INFINITY;

    for i in 0..interval_data.len() {
        for j in (i + 1)..interval_data.len() {
            let (x1, y1) = &interval_data[i];
            let (x2, y2) = &interval_data[j];

            if (x2 - x1).abs() < 1e-10 {
                continue;
            }

            // Slopes from lower-lower and upper-upper
            let s_ll = (y2.lower - y1.lower) / (x2 - x1);
            let s_uu = (y2.upper - y1.upper) / (x2 - x1);
            let s_lu = (y2.lower - y1.upper) / (x2 - x1);
            let s_ul = (y2.upper - y1.lower) / (x2 - x1);

            slope_lower = slope_lower.max(s_ll.min(s_uu).min(s_lu).min(s_ul));
            slope_upper = slope_upper.min(s_ll.max(s_uu).max(s_lu).max(s_ul));
        }
    }

    // Use OLS slope if robust bounds are too wide or invalid
    if !slope_lower.is_finite() || !slope_upper.is_finite() || slope_lower > slope_upper {
        return Some(ols);
    }

    let slope_interval = Interval::new(slope_lower, slope_upper);
    let slope = slope_interval.midpoint();

    // Compute intercept bounds
    let mut intercept_lower = f64::NEG_INFINITY;
    let mut intercept_upper = f64::INFINITY;

    for (x, y) in &interval_data {
        // y = slope * x + intercept => intercept = y - slope * x
        let int_lower = y.lower - slope_upper * x;
        let int_upper = y.upper - slope_lower * x;
        intercept_lower = intercept_lower.max(int_lower);
        intercept_upper = intercept_upper.min(int_upper);
    }

    let intercept_interval = if intercept_lower <= intercept_upper {
        Interval::new(intercept_lower, intercept_upper)
    } else {
        ols.intercept_interval
    };
    let intercept = intercept_interval.midpoint();

    Some(IntervalRegression {
        slope,
        intercept,
        slope_interval,
        intercept_interval,
        r_squared: ols.r_squared,
        data_points: ols.data_points,
        interval_bands: ols.interval_bands,
    })
}

/// Relative Jaccard-based regression weights.
/// Used for aggregating multiple estimates with different confidence levels.
pub fn compute_jix_weights(analyses: &[InversionAnalysis]) -> Vec<f64> {
    analyses
        .iter()
        .map(|a| {
            // Use the maximum Jaccard index from the curve as a confidence weight
            a.jaccard_curve
                .iter()
                .map(|(_, ji)| *ji)
                .fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Weighted interval regression using J_IX weights.
pub fn weighted_interval_regression(analyses: &[InversionAnalysis]) -> Option<IntervalRegression> {
    if analyses.len() < 2 {
        return None;
    }

    let weights = compute_jix_weights(analyses);
    let total_weight: f64 = weights.iter().sum();

    if total_weight <= 0.0 {
        return interval_least_squares(analyses);
    }

    // Weighted means
    let x_mean: f64 = analyses
        .iter()
        .zip(&weights)
        .map(|(a, w)| a.bt_ip_ratio * w)
        .sum::<f64>()
        / total_weight;

    let y_mean: f64 = analyses
        .iter()
        .zip(&weights)
        .map(|(a, w)| a.r_inv_point * w)
        .sum::<f64>()
        / total_weight;

    // Weighted sums
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;

    for (a, w) in analyses.iter().zip(&weights) {
        let dx = a.bt_ip_ratio - x_mean;
        let dy = a.r_inv_point - y_mean;
        ss_xy += w * dx * dy;
        ss_xx += w * dx * dx;
    }

    if ss_xx == 0.0 {
        return interval_least_squares(analyses);
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    // Use robust method for intervals
    let robust = robust_interval_regression(analyses)?;

    Some(IntervalRegression {
        slope,
        intercept,
        slope_interval: robust.slope_interval,
        intercept_interval: robust.intercept_interval,
        r_squared: robust.r_squared,
        data_points: robust.data_points,
        interval_bands: robust.interval_bands,
    })
}
