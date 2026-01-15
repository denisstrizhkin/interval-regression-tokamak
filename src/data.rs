//! Data loading module for tokamak experimental data.
//!
//! Parses CSV files containing electron temperature records and
//! sawtooth oscillation timestamps from Globus-M2 experiments.

use crate::interval::Interval;
use encoding_rs::WINDOWS_1252;
use encoding_rs_io::DecodeReaderBytesBuilder;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

/// Record from normalised_export.csv containing temperature measurements.
#[derive(Debug, Clone)]
pub struct TemperatureRecord {
    pub shot_number: u32,
    pub time: f64,           // Timestamp in ms
    pub r_minus_r_lcfs: f64, // R - R_lcfs in mm (radial position relative to last closed flux surface)
    pub te_normalized: f64,  // T_e/<Te> (normalized electron temperature)
    pub ne_normalized: f64,  // n_e/<ne> (normalized electron density)
    pub i_p: f64,            // Plasma current in kA
    pub b_t: f64,            // Toroidal magnetic field in T
    pub r_absolute: f64,     // Absolute radius R in cm
}

/// Record from inversion_radius.csv containing sawtooth event data.
#[derive(Debug, Clone)]
pub struct InversionRecord {
    pub shot_number: u32,
    pub ts_time_before: f64,  // Thomson scattering time before sawtooth (ms)
    pub delay_before: f64,    // Delay before sawtooth
    pub amp_before: f64,      // Amplitude before sawtooth
    pub period_before: f64,   // Period before sawtooth
    pub ts_time_after: f64,   // Thomson scattering time after sawtooth (ms)
    pub delay_after: f64,     // Delay after sawtooth
    pub amp_after: f64,       // Amplitude after sawtooth
    pub period_after: f64,    // Period after sawtooth (ms)
    pub r_inv_reference: f64, // Reference inversion radius value from literature
}

/// Loaded experimental data grouped by shot and time.
#[derive(Debug)]
pub struct ExperimentalData {
    /// Temperature records grouped by (shot_number, time)
    pub temperature_profiles: HashMap<(u32, i64), Vec<TemperatureRecord>>,
    /// Inversion event records
    pub inversion_events: Vec<InversionRecord>,
    /// Unique shot numbers with their plasma parameters
    pub shot_parameters: HashMap<u32, ShotParameters>,
}

/// Plasma parameters for a single shot.
#[derive(Debug, Clone)]
pub struct ShotParameters {
    pub shot_number: u32,
    pub i_p: f64, // Plasma current in kA
    pub b_t: f64, // Toroidal magnetic field in T
}

impl ExperimentalData {
    /// Gets the B_T / I_P ratio for a given shot.
    pub fn get_bt_ip_ratio(&self, shot_number: u32) -> Option<f64> {
        self.shot_parameters
            .get(&shot_number)
            .map(|p| p.b_t / p.i_p)
    }
}

/// Loads temperature data from normalised_export.csv.
pub fn load_temperature_data(
    path: &str,
) -> Result<HashMap<(u32, i64), Vec<TemperatureRecord>>, Box<dyn Error>> {
    let file = File::open(path)?;
    // Use encoding decoder for ISO-8859-1/Windows-1252 files
    let transcoded = DecodeReaderBytesBuilder::new()
        .encoding(Some(WINDOWS_1252))
        .build(file);
    let reader = BufReader::new(transcoded);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(reader);

    let mut profiles: HashMap<(u32, i64), Vec<TemperatureRecord>> = HashMap::new();

    for result in csv_reader.records() {
        let record = result?;

        // Skip if we don't have enough fields
        if record.len() < 19 {
            continue;
        }

        // Parse the required fields
        let shot_number: u32 = match record.get(0) {
            Some(s) => s.trim().parse().unwrap_or(0),
            None => continue,
        };

        if shot_number == 0 {
            continue;
        }

        let time: f64 = record
            .get(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let r_minus_r_lcfs: f64 = record
            .get(2)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let te_normalized: f64 = record
            .get(3)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let ne_normalized: f64 = record
            .get(4)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let i_p: f64 = record
            .get(5)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let b_t: f64 = record
            .get(6)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let r_absolute: f64 = record
            .get(18)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let temp_record = TemperatureRecord {
            shot_number,
            time,
            r_minus_r_lcfs,
            te_normalized,
            ne_normalized,
            i_p,
            b_t,
            r_absolute,
        };

        // Use truncated time (to ms) as key
        let time_key = (time * 100.0).round() as i64;
        profiles
            .entry((shot_number, time_key))
            .or_insert_with(Vec::new)
            .push(temp_record);
    }

    Ok(profiles)
}

/// Loads inversion radius data from inversion_radius.csv.
pub fn load_inversion_data(path: &str) -> Result<Vec<InversionRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    // Use encoding decoder for ISO-8859-1/Windows-1252 files
    let transcoded = DecodeReaderBytesBuilder::new()
        .encoding(Some(WINDOWS_1252))
        .build(file);
    let reader = BufReader::new(transcoded);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(reader);

    let mut records = Vec::new();

    for result in csv_reader.records() {
        let record = result?;

        // Skip header-like rows or incomplete rows
        if record.len() < 10 {
            continue;
        }

        // Parse shot number (first field, may have trailing comma in header)
        let shot_str = record.get(0).unwrap_or("").trim().trim_end_matches(',');
        let shot_number: u32 = match shot_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Parse time values
        let ts_time_before: f64 = record
            .get(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let delay_before: f64 = record
            .get(2)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let amp_before: f64 = record
            .get(3)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let period_before: f64 = record
            .get(4)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let ts_time_after: f64 = record
            .get(5)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let delay_after: f64 = record
            .get(6)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let amp_after: f64 = record
            .get(7)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let period_after: f64 = record
            .get(8)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let r_inv_reference: f64 = record
            .get(9)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        records.push(InversionRecord {
            shot_number,
            ts_time_before,
            delay_before,
            amp_before,
            period_before,
            ts_time_after,
            delay_after,
            amp_after,
            period_after,
            r_inv_reference,
        });
    }

    Ok(records)
}

/// Extracts shot parameters from temperature data.
pub fn extract_shot_parameters(
    profiles: &HashMap<(u32, i64), Vec<TemperatureRecord>>,
) -> HashMap<u32, ShotParameters> {
    let mut params: HashMap<u32, ShotParameters> = HashMap::new();

    for ((shot, _), records) in profiles.iter() {
        if let Some(rec) = records.first() {
            params.entry(*shot).or_insert_with(|| ShotParameters {
                shot_number: *shot,
                i_p: rec.i_p,
                b_t: rec.b_t,
            });
        }
    }

    params
}

/// Loads all experimental data from the data directory.
pub fn load_all_data(data_dir: &str) -> Result<ExperimentalData, Box<dyn Error>> {
    let temp_path = format!("{}/normalised_export.csv", data_dir);
    let inv_path = format!("{}/inversion_radius.csv", data_dir);

    println!("Loading temperature data from {}...", temp_path);
    let temperature_profiles = load_temperature_data(&temp_path)?;
    println!(
        "  Loaded {} profile measurements",
        temperature_profiles.len()
    );

    println!("Loading inversion radius data from {}...", inv_path);
    let inversion_events = load_inversion_data(&inv_path)?;
    println!("  Loaded {} inversion events", inversion_events.len());

    let shot_parameters = extract_shot_parameters(&temperature_profiles);
    println!("  Extracted parameters for {} shots", shot_parameters.len());

    Ok(ExperimentalData {
        temperature_profiles,
        inversion_events,
        shot_parameters,
    })
}

/// Gets temperature profile for a specific shot and time.
/// Returns data points sorted by radius.
pub fn get_temperature_profile(
    data: &ExperimentalData,
    shot: u32,
    time: f64,
) -> Option<Vec<TemperatureRecord>> {
    let time_key = (time * 100.0).round() as i64;
    data.temperature_profiles
        .get(&(shot, time_key))
        .cloned()
        .map(|mut v| {
            v.sort_by(|a, b| a.r_absolute.partial_cmp(&b.r_absolute).unwrap());
            v
        })
}

/// Creates an interval for the temperature value with measurement uncertainty.
/// Uses a simple relative error model (e.g., 10% uncertainty).
pub fn temperature_with_uncertainty(te: f64, relative_error: f64) -> Interval {
    let error = te.abs() * relative_error;
    Interval::from_center_radius(te, error)
}
