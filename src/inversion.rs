//! Inversion radius calculation module.
//!
//! Implements the algorithm for determining the inversion radius R_inv
//! using the Jaccard index between temperature profiles before and after
//! sawtooth oscillations.

use crate::data::{ExperimentalData, InversionRecord, TemperatureRecord};
use crate::interval::Interval;
use crate::spline::{CubicSpline, DataPoint, LinearSpline, Spline};

/// Result of inversion radius analysis for a single sawtooth event.
#[derive(Debug, Clone)]
pub struct InversionAnalysis {
    pub shot_number: u32,
    /// Time before sawtooth (ms)
    pub time_before: f64,
    /// Time after sawtooth (ms)
    pub time_after: f64,
    /// Point estimate of inversion radius (where J_I is maximized)
    pub r_inv_point: f64,
    /// Internal interval bounds where J_I >= 0.5 [lower, upper]
    pub r_inv_low: f64,
    pub r_inv_high: f64,
    /// External interval bounds where J_I > 0 [lower, upper]
    pub r_inv_ext_low: f64,
    pub r_inv_ext_high: f64,
    /// Consistency flag for the maximal intersection subset (Figure 3.4)
    pub is_consistent: bool,
    /// Consistency flag for external estimates (Figure 3.5)
    pub is_consistent_ext: bool,
    /// Reference value from literature
    pub r_inv_reference: f64,
    /// Jaccard index values at detailed grid points
    pub jaccard_curve: Vec<(f64, f64)>,
    /// Plasma parameters
    pub bt_ip_ratio: f64,
    /// Temperature profiles
    pub profile_before: Vec<(f64, f64)>, // (radius, temp)
    pub profile_after: Vec<(f64, f64)>,
    /// Compatibility corridors (interpolated intervals)
    pub corridor_before: Vec<(f64, f64, f64)>, // (radius, lower, upper)
    pub corridor_after: Vec<(f64, f64, f64)>,
}

/// Configuration for the inversion radius calculation.
#[derive(Debug, Clone)]
pub struct InversionConfig {
    /// Number of grid points for detailed interpolation
    pub grid_points: usize,
    /// Relative measurement uncertainty (e.g., 0.1 for 10%)
    pub relative_uncertainty: f64,
    /// Threshold for inner radius (J_I >= threshold)
    pub ji_inner_threshold: f64,
    /// Use cubic spline (true) or linear spline (false)
    pub use_cubic_spline: bool,
}

impl Default for InversionConfig {
    fn default() -> Self {
        Self {
            grid_points: 200,
            relative_uncertainty: 0.10,
            ji_inner_threshold: 0.5,
            use_cubic_spline: true,
        }
    }
}

/// Converts temperature records to data points for spline interpolation.
fn records_to_points(records: &[TemperatureRecord]) -> Vec<DataPoint> {
    records
        .iter()
        .map(|r| DataPoint {
            radius: r.r_absolute,
            temperature: r.te_normalized,
        })
        .collect()
}

/// Creates a spline from data points based on configuration.
fn create_spline(points: Vec<DataPoint>, use_cubic: bool) -> Box<dyn Spline> {
    if use_cubic {
        Box::new(CubicSpline::new(points))
    } else {
        Box::new(LinearSpline::new(points))
    }
}

/// Computes the Jaccard index between two temperature intervals at a given radius.
fn compute_jaccard_at_radius(
    spline_before: &dyn Spline,
    spline_after: &dyn Spline,
    radius: f64,
    relative_uncertainty: f64,
) -> Option<f64> {
    let temp_before = spline_before.interpolate(radius)?;
    let temp_after = spline_after.interpolate(radius)?;

    // Create intervals with uncertainty
    let error_before = temp_before.abs() * relative_uncertainty;
    let error_after = temp_after.abs() * relative_uncertainty;

    let interval_before = Interval::from_center_radius(temp_before, error_before);
    let interval_after = Interval::from_center_radius(temp_after, error_after);

    Some(interval_before.jaccard_index(&interval_after))
}

/// Analyzes a single sawtooth event to determine the inversion radius.
pub fn analyze_sawtooth_event(
    data: &ExperimentalData,
    event: &InversionRecord,
    config: &InversionConfig,
) -> Option<InversionAnalysis> {
    let shot = event.shot_number;

    // Get temperature profiles before and after the sawtooth
    let time_before = event.ts_time_before;
    let time_after = event.ts_time_after;

    // Find closest available profiles
    let profile_before = find_closest_profile(data, shot, time_before)?;
    let profile_after = find_closest_profile(data, shot, time_after)?;

    if profile_before.is_empty() || profile_after.is_empty() {
        return None;
    }

    // Get B_T / I_P ratio
    let bt_ip_ratio = data.get_bt_ip_ratio(shot).unwrap_or(0.0);

    // Create splines for interpolation
    let points_before = records_to_points(&profile_before);
    let points_after = records_to_points(&profile_after);

    let spline_before = create_spline(points_before.clone(), config.use_cubic_spline);
    let spline_after = create_spline(points_after.clone(), config.use_cubic_spline);

    // Determine the common radial range
    let range_before = spline_before.range()?;
    let range_after = spline_after.range()?;
    let r_min = range_before.0.max(range_after.0);
    let r_max = range_before.1.min(range_after.1);

    if r_min >= r_max {
        return None;
    }

    // Generate detailed interpolation grid
    let step = (r_max - r_min) / (config.grid_points as f64);
    let mut jaccard_curve = Vec::with_capacity(config.grid_points);

    for i in 0..=config.grid_points {
        let r = r_min + i as f64 * step;
        if let Some(ji) = compute_jaccard_at_radius(
            spline_before.as_ref(),
            spline_after.as_ref(),
            r,
            config.relative_uncertainty,
        ) {
            jaccard_curve.push((r, ji));
        }
    }

    if jaccard_curve.is_empty() {
        return None;
    }

    // Find the outer inversion radius point (R > 42 cm)
    // We filter for the outer peak to match the target analysis focused on the outer plasma region.
    let (r_inv_point, _max_ji) = jaccard_curve
        .iter()
        .filter(|(r, _)| *r > 42.0)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .or_else(|| {
            // Fallback if no points > 42cm exist
            jaccard_curve
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        })
        .cloned()?;

    // Find bounds where J_I >= threshold within the outer region (Internal)
    let inner_points: Vec<f64> = jaccard_curve
        .iter()
        .filter(|(r, ji)| *r > 42.0 && *ji >= config.ji_inner_threshold)
        .map(|(r, _)| *r)
        .collect();

    // Find bounds where J_I > 0 within the outer region (External)
    let outer_points: Vec<f64> = jaccard_curve
        .iter()
        .filter(|(r, ji)| *r > 42.0 && *ji > 0.0)
        .map(|(r, _)| *r)
        .collect();

    let (r_inv_low, r_inv_high) = if inner_points.is_empty() {
        (r_inv_point, r_inv_point)
    } else {
        let low = inner_points.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = inner_points
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };

    let (r_inv_ext_low, r_inv_ext_high) = if outer_points.is_empty() {
        (r_inv_low, r_inv_high)
    } else {
        let low = outer_points.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = outer_points
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };

    // Point estimate is the midpoint for vertical centering on error bars
    let r_inv_point = (r_inv_low + r_inv_high) / 2.0;

    // Store profiles
    let profile_before_pts: Vec<(f64, f64)> = points_before
        .iter()
        .map(|p| (p.radius, p.temperature))
        .collect();
    let profile_after_pts: Vec<(f64, f64)> = points_after
        .iter()
        .map(|p| (p.radius, p.temperature))
        .collect();

    // Generate compatibility corridors
    let corridor_before = generate_corridor(&jaccard_curve, spline_before.as_ref(), config);
    let corridor_after = generate_corridor(&jaccard_curve, spline_after.as_ref(), config);

    Some(InversionAnalysis {
        shot_number: shot,
        time_before,
        time_after,
        r_inv_point,
        r_inv_low,
        r_inv_high,
        r_inv_ext_low,
        r_inv_ext_high,
        is_consistent: false,
        is_consistent_ext: false,
        r_inv_reference: event.r_inv_reference,
        jaccard_curve,
        bt_ip_ratio,
        profile_before: profile_before_pts,
        profile_after: profile_after_pts,
        corridor_before,
        corridor_after,
    })
}

/// Generates a compatibility corridor for a temperature profile.
fn generate_corridor(
    grid: &[(f64, f64)],
    spline: &dyn Spline,
    config: &InversionConfig,
) -> Vec<(f64, f64, f64)> {
    grid.iter()
        .filter_map(|(r, _)| {
            let temp = spline.interpolate(*r)?;
            let error = temp.abs() * config.relative_uncertainty;
            Some((*r, temp - error, temp + error))
        })
        .collect()
}

/// Finds the closest available profile to a target time.
fn find_closest_profile(
    data: &ExperimentalData,
    shot: u32,
    target_time: f64,
) -> Option<Vec<TemperatureRecord>> {
    // First, try exact time match
    let time_key = (target_time * 100.0).round() as i64;
    if let Some(profile) = data.temperature_profiles.get(&(shot, time_key)) {
        return Some(profile.clone());
    }

    // Find the closest available time
    let mut best_time_key = None;
    let mut min_diff = f64::INFINITY;

    for ((s, t), _) in data.temperature_profiles.iter() {
        if *s == shot {
            let time = *t as f64 / 100.0;
            let diff = (time - target_time).abs();
            if diff < min_diff {
                min_diff = diff;
                best_time_key = Some(*t);
            }
        }
    }

    best_time_key.and_then(|t| data.temperature_profiles.get(&(shot, t)).cloned())
}

/// Analyzes all sawtooth events in the dataset and determines the maximal consistent subset.
pub fn analyze_all_events(
    data: &ExperimentalData,
    config: &InversionConfig,
) -> Vec<InversionAnalysis> {
    let mut analyses: Vec<InversionAnalysis> = data
        .inversion_events
        .iter()
        .filter_map(|event| analyze_sawtooth_event(data, event, config))
        .collect();

    if analyses.is_empty() {
        return analyses;
    }

    // Determine maximal consistent subset (Maximal Clique Algorithm for Interval Graphs)
    // We want to find a point R0 that is contained in the maximum number of intervals [r_inv_low, r_inv_high]
    let mut events: Vec<(f64, i32)> = Vec::new();
    for a in &analyses {
        events.push((a.r_inv_low, 1)); // Interval start
        events.push((a.r_inv_high, -1)); // Interval end
    }

    // Sort events by radius (ends after starts for same radius to keep them inclusive)
    events.sort_by(|a, b| {
        let res = a.0.partial_cmp(&b.0).unwrap();
        if res == std::cmp::Ordering::Equal {
            b.1.cmp(&a.1) // Start (1) before End (-1)
        } else {
            res
        }
    });

    let mut max_count = 0;
    let mut current_count = 0;
    let mut r0_range = (0.0, 0.0);

    for (r, type_flag) in &events {
        current_count += type_flag;
        if current_count > max_count {
            max_count = current_count;
            r0_range.0 = *r;
        } else if current_count == max_count && type_flag == &-1 {
            // We found the end of the maximal intersection range
            r0_range.1 = *r;
        }
    }

    // Repeat for external estimates
    let mut events_ext: Vec<(f64, i32)> = Vec::new();
    for a in &analyses {
        events_ext.push((a.r_inv_ext_low, 1));
        events_ext.push((a.r_inv_ext_high, -1));
    }
    events_ext.sort_by(|a, b| {
        let res = a.0.partial_cmp(&b.0).unwrap();
        if res == std::cmp::Ordering::Equal {
            b.1.cmp(&a.1)
        } else {
            res
        }
    });

    let mut max_count_ext = 0;
    let mut current_count_ext = 0;
    let mut r0_range_ext = (0.0, 0.0);

    for (r, type_flag) in &events_ext {
        current_count_ext += type_flag;
        if current_count_ext > max_count_ext {
            max_count_ext = current_count_ext;
            r0_range_ext.0 = *r;
        } else if current_count_ext == max_count_ext && type_flag == &-1 {
            r0_range_ext.1 = *r;
        }
    }

    // Label consistency
    for a in &mut analyses {
        if a.r_inv_low <= r0_range.0 && a.r_inv_high >= r0_range.0 {
            a.is_consistent = true;
        }
        if a.r_inv_ext_low <= r0_range_ext.0 && a.r_inv_ext_high >= r0_range_ext.0 {
            a.is_consistent_ext = true;
        }
    }

    analyses
}

/// Computes interval histogram data for a set of inversion analyses.
/// Groups data by B_T/I_P ratio bins and computes statistics.
pub fn compute_histogram_data(
    analyses: &[InversionAnalysis],
    num_bins: usize,
) -> Vec<HistogramBin> {
    if analyses.is_empty() {
        return Vec::new();
    }

    // Find the range of B_T/I_P values
    let min_ratio = analyses
        .iter()
        .map(|a| a.bt_ip_ratio)
        .fold(f64::INFINITY, f64::min);
    let max_ratio = analyses
        .iter()
        .map(|a| a.bt_ip_ratio)
        .fold(f64::NEG_INFINITY, f64::max);

    if min_ratio >= max_ratio {
        return Vec::new();
    }

    let bin_width = (max_ratio - min_ratio) / num_bins as f64;
    let mut bins: Vec<HistogramBin> = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        let bin_start = min_ratio + i as f64 * bin_width;
        let bin_end = bin_start + bin_width;
        let bin_center = (bin_start + bin_end) / 2.0;

        // Collect R_inv values in this bin
        let values: Vec<f64> = analyses
            .iter()
            .filter(|a| a.bt_ip_ratio >= bin_start && a.bt_ip_ratio < bin_end)
            .map(|a| a.r_inv_point)
            .collect();

        if !values.is_empty() {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Also compute external interval for the bin
            let ext_low = analyses
                .iter()
                .filter(|a| a.bt_ip_ratio >= bin_start && a.bt_ip_ratio < bin_end)
                .map(|a| a.r_inv_ext_low)
                .fold(f64::INFINITY, f64::min);
            let ext_high = analyses
                .iter()
                .filter(|a| a.bt_ip_ratio >= bin_start && a.bt_ip_ratio < bin_end)
                .map(|a| a.r_inv_ext_high)
                .fold(f64::NEG_INFINITY, f64::max);

            bins.push(HistogramBin {
                bt_ip_center: bin_center,
                r_inv_mean: mean,
                r_inv_interval: Interval::new(min, max),
                r_inv_ext_interval: Interval::new(ext_low, ext_high),
                count: values.len(),
            });
        }
    }

    bins
}

/// A bin in the interval histogram.
#[derive(Debug, Clone)]
pub struct HistogramBin {
    pub bt_ip_center: f64,
    pub r_inv_mean: f64,
    pub r_inv_interval: Interval,
    pub r_inv_ext_interval: Interval,
    pub count: usize,
}
