//! SIMD-optimized operations for event camera data processing
//!
//! This module provides vectorized implementations of common operations
//! used in event clustering, including:
//! - Coordinate conversions
//! - Distance calculations
//! - Statistical computations (mean, variance)
//! - Hot pixel filtering
//!
//! These optimizations use Arm Neon intrinsics when available, with
//! fallback scalar implementations for other architectures.

/// Calculate mean of f64 slice using SIMD when available
#[inline]
pub fn calculate_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && values.len() >= 4 {
            return unsafe { calculate_mean_neon(values) };
        }
    }

    // Scalar fallback
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn calculate_mean_neon(values: &[f64]) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f64(0.0);
    let chunks = values.len() / 2;
    let remainder = values.len() % 2;

    // Process 2 f64 values at a time (128-bit Neon register)
    for i in 0..chunks {
        let v = vld1q_f64(&values[i * 2]);
        sum = vaddq_f64(sum, v);
    }

    // Horizontal add to get final sum
    let mut total = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

    // Handle remainder
    if remainder > 0 {
        total += values[chunks * 2];
    }

    total / values.len() as f64
}

/// Calculate variance of f64 slice given the mean, using SIMD when available
#[inline]
pub fn calculate_variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && values.len() >= 4 {
            return unsafe { calculate_variance_neon(values, mean) };
        }
    }

    // Scalar fallback
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn calculate_variance_neon(values: &[f64], mean: f64) -> f64 {
    use std::arch::aarch64::*;

    let vmean = vdupq_n_f64(mean);
    let mut vsum = vdupq_n_f64(0.0);
    let chunks = values.len() / 2;
    let remainder = values.len() % 2;

    // Process 2 f64 values at a time
    for i in 0..chunks {
        let v = vld1q_f64(&values[i * 2]);
        let diff = vsubq_f64(v, vmean);
        let squared = vmulq_f64(diff, diff);
        vsum = vaddq_f64(vsum, squared);
    }

    // Horizontal add to get final sum
    let mut total = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);

    // Handle remainder
    if remainder > 0 {
        let last = values[chunks * 2] - mean;
        total += last * last;
    }

    total / values.len() as f64
}

/// Calculate standard deviation (square root of variance)
#[inline]
pub fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    calculate_variance(values, mean).sqrt()
}

/// Calculate mean and standard deviation in one pass
#[inline]
pub fn calculate_mean_and_std(values: &[f64]) -> (f64, f64) {
    let mean = calculate_mean(values);
    let std = calculate_std_dev(values, mean);
    (mean, std)
}

/// Convert u16 coordinates to f64 using SIMD when available
#[inline]
pub fn convert_coords_u16_to_f64(coords: &[u16], output: &mut [f64]) {
    assert_eq!(
        coords.len(),
        output.len(),
        "Input and output must have same length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && coords.len() >= 8 {
            return unsafe { convert_coords_u16_to_f64_neon(coords, output) };
        }
    }

    // Scalar fallback
    for (i, &coord) in coords.iter().enumerate() {
        output[i] = coord as f64;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_coords_u16_to_f64_neon(coords: &[u16], output: &mut [f64]) {
    // Note: Converting u16 to f64 with Neon is complex and may not provide
    // significant performance benefit. Using scalar fallback for simplicity.
    // For optimal performance, consider restructuring to use f32 instead.
    for (i, &coord) in coords.iter().enumerate() {
        output[i] = coord as f64;
    }
}

// Note: Event-specific functions are commented out as Event type is in main.rs
// Uncomment and adapt if Event type is moved to library

// /// Extract X coordinates from events using SIMD when available
// #[inline]
// pub fn extract_x_coords(events: &[Event], output: &mut [f64]) {
//     // Scalar fallback
//     for (i, event) in events.iter().enumerate() {
//         output[i] = event.x as f64;
//     }
// }
//
// /// Extract Y coordinates from events using SIMD when available
// #[inline]
// pub fn extract_y_coords(events: &[Event], output: &mut [f64]) {
//     // Scalar implementation
//     for (i, event) in events.iter().enumerate() {
//         output[i] = event.y as f64;
//     }
// }

/// Calculate sum of squared differences between two slices using SIMD
#[inline]
pub fn sum_squared_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have same length");

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && a.len() >= 4 {
            return unsafe { sum_squared_diff_neon(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_squared_diff_neon(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::*;

    let mut vsum = vdupq_n_f64(0.0);
    let chunks = a.len() / 2;
    let remainder = a.len() % 2;

    // Process 2 f64 values at a time
    for i in 0..chunks {
        let va = vld1q_f64(&a[i * 2]);
        let vb = vld1q_f64(&b[i * 2]);
        let diff = vsubq_f64(va, vb);
        let squared = vmulq_f64(diff, diff);
        vsum = vaddq_f64(vsum, squared);
    }

    // Horizontal add
    let mut total = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);

    // Handle remainder
    if remainder > 0 {
        let diff = a[chunks * 2] - b[chunks * 2];
        total += diff * diff;
    }

    total
}

/// Calculate Euclidean distance between two 2D points using SIMD
#[inline]
pub fn euclidean_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    (dx * dx + dy * dy).sqrt()
}

/// Batch calculate distances from a point to multiple points
#[inline]
pub fn batch_distances(x: f64, y: f64, target_x: &[f64], target_y: &[f64], distances: &mut [f64]) {
    assert_eq!(target_x.len(), target_y.len());
    assert_eq!(target_x.len(), distances.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && target_x.len() >= 4 {
            return unsafe { batch_distances_neon(x, y, target_x, target_y, distances) };
        }
    }

    // Scalar fallback
    for i in 0..target_x.len() {
        let dx = x - target_x[i];
        let dy = y - target_y[i];
        distances[i] = (dx * dx + dy * dy).sqrt();
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn batch_distances_neon(
    x: f64,
    y: f64,
    target_x: &[f64],
    target_y: &[f64],
    distances: &mut [f64],
) {
    use std::arch::aarch64::*;

    let vx = vdupq_n_f64(x);
    let vy = vdupq_n_f64(y);
    let chunks = target_x.len() / 2;
    let remainder = target_x.len() % 2;

    // Process 2 points at a time
    for i in 0..chunks {
        let tx = vld1q_f64(&target_x[i * 2]);
        let ty = vld1q_f64(&target_y[i * 2]);

        let dx = vsubq_f64(vx, tx);
        let dy = vsubq_f64(vy, ty);

        let dx_sq = vmulq_f64(dx, dx);
        let dy_sq = vmulq_f64(dy, dy);

        let dist_sq = vaddq_f64(dx_sq, dy_sq);

        // Extract and compute square root (no direct vsqrtq_f64 in Neon, use scalar)
        distances[i * 2] = vgetq_lane_f64(dist_sq, 0).sqrt();
        distances[i * 2 + 1] = vgetq_lane_f64(dist_sq, 1).sqrt();
    }

    // Handle remainder
    if remainder > 0 {
        let i = chunks * 2;
        let dx = x - target_x[i];
        let dy = y - target_y[i];
        distances[i] = (dx * dx + dy * dy).sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = calculate_mean(&values);
        assert!((mean - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_variance() {
        let values = vec![2.0, 4.0, 6.0, 8.0];
        let mean = 5.0;
        let variance = calculate_variance(&values, mean);
        // Expected: ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9+1+1+9)/4 = 5.0
        assert!((variance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_mean_and_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = calculate_mean_and_std(&values);
        assert!((mean - 5.0).abs() < 1e-10);
        assert!(std > 0.0);
    }

    #[test]
    fn test_sum_squared_diff() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let ssd = sum_squared_diff(&a, &b);
        // Expected: (1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2 = 1+1+1+1 = 4.0
        assert!((ssd - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_2d() {
        let dist = euclidean_distance_2d(0.0, 0.0, 3.0, 4.0);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_distances() {
        let x = 0.0;
        let y = 0.0;
        let target_x = vec![3.0, 5.0, 12.0, 8.0];
        let target_y = vec![4.0, 12.0, 5.0, 15.0];
        let mut distances = vec![0.0; 4];

        batch_distances(x, y, &target_x, &target_y, &mut distances);

        assert!((distances[0] - 5.0).abs() < 1e-10);
        assert!((distances[1] - 13.0).abs() < 1e-10);
        assert!((distances[2] - 13.0).abs() < 1e-10);
        assert!((distances[3] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_convert_coords_u16_to_f64() {
        let coords = vec![10u16, 20, 30, 40, 50, 60, 70, 80];
        let mut output = vec![0.0; 8];

        convert_coords_u16_to_f64(&coords, &mut output);

        for i in 0..8 {
            assert_eq!(output[i], coords[i] as f64);
        }
    }
}
