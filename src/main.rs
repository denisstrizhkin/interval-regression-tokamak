//! Globus-M2 Tokamak Inversion Radius Analysis Suite
//!
//! This program analyzes sawtooth oscillations in tokamak plasma to determine
//! the inversion radius using interval-valued statistics and spline interpolation.
//!
//! Based on research from Bazhenov presentation and Mordovin thesis on
//! interval regression methods for plasma physics.
//!
//! # Usage
//! ```
//! cargo run --release
//! gnuplot plot_results.gp
//! ```
//!
//! # Algorithm Steps
//! 1. Load experimental data (temperature profiles and sawtooth events)
//! 2. For each sawtooth event:
//!    - Interpolate T_before and T_after profiles using splines
//!    - Compute Jaccard index J_I across the radial grid
//!    - Determine R_inv (point estimate) where J_I is maximized
//!    - Compute twin estimates (R_inn: J_I >= 0.5, R_out: J_I > 0)
//! 3. Perform interval regression of R_inv on B_T/I_P
//! 4. Generate forecasts for higher B_T/I_P values
//! 5. Export results for gnuplot visualization

mod data;
mod interval;
mod inversion;
mod output;
mod regression;
mod spline;

use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Globus-M2 Tokamak Inversion Radius Analysis Suite                 ║");
    println!("║  Interval Regression and Jaccard Index Method                      ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // Determine data directory
    let current_dir = env::current_dir()?;
    let data_dir = format!("{}/data", current_dir.display());
    let output_dir = current_dir.to_string_lossy().to_string();

    // Step 1: Load experimental data
    println!("Step 1: Loading experimental data...");
    let data = data::load_all_data(&data_dir)?;

    println!(
        "  Found {} unique shots with plasma parameters",
        data.shot_parameters.len()
    );
    println!(
        "  Found {} sawtooth inversion events",
        data.inversion_events.len()
    );
    println!();

    // Step 2: Configure the analysis
    println!("Step 2: Configuring analysis parameters...");
    let config = inversion::InversionConfig {
        grid_points: 200,
        relative_uncertainty: 0.10, // 10% measurement uncertainty
        ji_inner_threshold: 0.5,    // J_I >= 0.5 for inner radius
        use_cubic_spline: true,     // Use cubic splines for smooth profiles
    };

    println!("  Grid points: {}", config.grid_points);
    println!(
        "  Relative uncertainty: {:.0}%",
        config.relative_uncertainty * 100.0
    );
    println!(
        "  Inner radius J_I threshold: {:.1}",
        config.ji_inner_threshold
    );
    println!(
        "  Spline type: {}",
        if config.use_cubic_spline {
            "Cubic (3rd order)"
        } else {
            "Linear (1st order)"
        }
    );
    println!();

    // Step 3: Analyze all sawtooth events
    println!("Step 3: Analyzing sawtooth events...");
    let analyses = inversion::analyze_all_events(&data, &config);

    println!("  Successfully analyzed {} sawtooth events", analyses.len());

    if analyses.is_empty() {
        println!("Warning: No successful analyses. Check data alignment.");
        return Ok(());
    }

    // Print summary statistics
    let r_inv_points: Vec<f64> = analyses.iter().map(|a| a.r_inv_point).collect();
    let r_inv_mean = r_inv_points.iter().sum::<f64>() / r_inv_points.len() as f64;
    let r_inv_min = r_inv_points.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_inv_max = r_inv_points
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  R_inv statistics:");
    println!("    Mean: {:.2} cm", r_inv_mean);
    println!("    Range: [{:.2}, {:.2}] cm", r_inv_min, r_inv_max);
    println!();

    // Step 4: Perform interval regression
    println!("Step 4: Performing interval regression...");

    let regression = regression::weighted_interval_regression(&analyses);

    if let Some(ref reg) = regression {
        println!(
            "  Linear regression: R_inv = {:.4} * (B_T/I_P) + {:.2}",
            reg.slope, reg.intercept
        );
        println!(
            "  Slope interval: [{:.4}, {:.4}]",
            reg.slope_interval.lower, reg.slope_interval.upper
        );
        println!(
            "  Intercept interval: [{:.2}, {:.2}]",
            reg.intercept_interval.lower, reg.intercept_interval.upper
        );
        println!("  R-squared: {:.4}", reg.r_squared);
    } else {
        println!("  Warning: Regression could not be computed");
    }
    println!();

    // Step 5: Generate forecasts
    println!("Step 5: Generating forecasts...");

    let forecast = if let Some(ref reg) = regression {
        // Find B_T/I_P range from data
        let bt_ip_values: Vec<f64> = analyses.iter().map(|a| a.bt_ip_ratio).collect();
        let bt_ip_min = bt_ip_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let bt_ip_max = bt_ip_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Extend forecast range by 50% beyond observed data
        let forecast_min = bt_ip_min * 0.8;
        let forecast_max = bt_ip_max * 1.5;

        println!(
            "  Observed B_T/I_P range: [{:.5}, {:.5}] T/kA",
            bt_ip_min, bt_ip_max
        );
        println!(
            "  Forecast range: [{:.5}, {:.5}] T/kA",
            forecast_min, forecast_max
        );

        let forecast = reg.forecast(forecast_min, forecast_max, 100);

        // Show some forecast points
        if forecast.len() >= 3 {
            println!("  Forecast examples:");
            for fp in forecast.iter().step_by(25) {
                println!(
                    "    B_T/I_P = {:.5}: R_inv = {:.2} [{:.2}, {:.2}] cm",
                    fp.bt_ip, fp.r_inv_point, fp.r_inv_lower, fp.r_inv_upper
                );
            }
        }

        forecast
    } else {
        Vec::new()
    };
    println!();

    // Step 6: Compute interval histogram data
    println!("Step 6: Computing interval histogram data...");
    let histogram_bins = inversion::compute_histogram_data(&analyses, 10);

    println!("  Generated {} histogram bins", histogram_bins.len());
    for bin in histogram_bins.iter().take(5) {
        println!(
            "    B_T/I_P = {:.5}: R_inv = {:.2} ± {:.2} cm (n={})",
            bin.bt_ip_center,
            bin.r_inv_mean,
            bin.r_inv_interval.radius(),
            bin.count
        );
    }
    println!();

    // Step 7: Export results
    println!("Step 7: Exporting results...");
    output::export_results(&analyses, regression.as_ref(), &forecast, &output_dir)?;
    output::generate_gnuplot_script(&output_dir)?;
    println!();

    // Final summary
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Analysis Complete                                                 ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║  Run 'gnuplot plot_results.gp' to generate visualization plots    ║");
    println!("║  Output files:                                                     ║");
    println!("║    - report/traces/plot_data.csv       (main analysis results)      ║");
    println!("║    - report/traces/jaccard_curves.csv  (Jaccard distributions)     ║");
    println!("║    - report/traces/profiles.csv        (temperature profiles)      ║");
    println!("║    - report/traces/regression_data.csv (regression forecasts)    ║");
    println!("║    - plot_results.gp                  (gnuplot script)             ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basics() {
        let a = interval::Interval::new(1.0, 3.0);
        let b = interval::Interval::new(2.0, 4.0);

        assert_eq!(a.width(), 2.0);
        assert_eq!(a.midpoint(), 2.0);

        let intersection = a.intersection(&b).unwrap();
        assert_eq!(intersection.lower, 2.0);
        assert_eq!(intersection.upper, 3.0);

        let ji = a.jaccard_index(&b);
        // Intersection width = 1, Union hull width = 3
        assert!((ji - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = interval::Interval::new(1.0, 2.0);
        let b = interval::Interval::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.lower, 4.0);
        assert_eq!(sum.upper, 6.0);

        let diff = b - a;
        assert_eq!(diff.lower, 1.0);
        assert_eq!(diff.upper, 3.0);

        let prod = a * b;
        assert_eq!(prod.lower, 3.0);
        assert_eq!(prod.upper, 8.0);
    }

    #[test]
    fn test_spline_interpolation() {
        use spline::{CubicSpline, DataPoint, LinearSpline};

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
        ];

        let linear = LinearSpline::new(points.clone());
        let cubic = CubicSpline::new(points);

        // At exact points, both should match
        assert!((linear.interpolate(1.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((cubic.interpolate(1.0).unwrap() - 1.0).abs() < 1e-10);

        // Linear interpolation at midpoint
        let mid_linear = linear.interpolate(0.5).unwrap();
        assert!((mid_linear - 0.5).abs() < 1e-10);
    }
}
