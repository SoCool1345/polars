// Some formulae:
//     mean_x = sum(weight[i] * x[i]) / sum(weight)
//     dp_xy = weighted sum of deviation products of variables x, y, written in
//             the paper as simply XY.
//     dp_xy = sum(weight[i] * (x[i] - mean_x) * (y[i] - mean_y))
//
//     cov(x, y) = dp_xy / sum(weight)
//     var(x) = cov(x, x)
//
// Algorithms from:
// Numerically stable parallel computation of (co-)variance.
// Schubert, E. & Gertz, M. (2018).
//
// Key equations from the paper:
// (17) for mean update, (23) for dp update (and also Table 1).
//
//
// For higher moments we refer to:
// Numerically Stable, Scalable Formulas for Parallel and Online Computation of
// Higher-Order Multivariate Central Moments with Arbitrary Weights.
// Pébay, P. & Terriberry, T. B. & Kolla, H. & Bennett J. (2016)
//
// Key equations from paper:
// (3.26) mean update, (3.27) moment update.
//
// Here we use mk to mean the weighted kth central moment:
//    mk = sum(weight[i] * (x[i] - mean_x)**k)
// Note that we'll use the terms m2 = dp = dp_xx if unambiguous.

#![allow(clippy::collapsible_else_if)]

use num_traits::AsPrimitive;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::types::NativeType;
use polars_utils::algebraic_ops::*;

const CHUNK_SIZE: usize = 128;

/// Arithmetic mean of `x`, anchored on its first element, so that the result
/// does not depend on the summation order (the compiler may reassociate or
/// vectorize the sum, and callers combine several chunks). Without an anchor a
/// constant column can end up with a mean a few ulp off `x[0]`, which makes the
/// deviations tiny but non-zero and turns `var`/`cov`/`corr` into ratios of
/// such values. Here every deviation of a constant column is exactly `0.0`.
#[inline]
fn anchored_mean(x: &[f64], weight: f64) -> f64 {
    let anchor = x[0];
    anchor + alg_sum_f64(x.iter().map(|&xi| xi - anchor)) / weight
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct VarState {
    weight: f64,
    mean: f64,
    dp: f64,
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct CovState {
    weight: f64,
    mean_x: f64,
    mean_y: f64,
    dp_xy: f64,
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct PearsonState {
    weight: f64,
    mean_x: f64,
    mean_y: f64,
    dp_xx: f64,
    dp_xy: f64,
    dp_yy: f64,
}

impl VarState {
    fn new(x: &[f64]) -> Self {
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let mean = anchored_mean(x, weight);
        Self {
            weight,
            mean,
            dp: alg_sum_f64(x.iter().map(|&xi| (xi - mean) * (xi - mean))),
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.dp = 0.0;
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Just a specialized version of
        // self.combine(&Self { weight: 1.0, mean: x, dp: 0.0 })
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let new_mean = self.mean + delta_mean / new_weight;
        self.dp += (x - new_mean) * delta_mean;
        self.weight = new_weight;
        self.mean = new_mean;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean = other.mean - self.mean;
        let new_mean = self.mean + delta_mean * other_weight_frac;
        self.dp += other.dp + other.weight * (other.mean - new_mean) * delta_mean;
        self.weight = new_weight;
        self.mean = new_mean;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, ddof: u8) -> Option<f64> {
        if self.weight <= ddof as f64 {
            None
        } else {
            let var = self.dp / (self.weight - ddof as f64);
            Some(if var < 0.0 {
                // Variance can't be negative, except through numerical instability.
                // We don't use f64::max here so we propagate nans.
                0.0
            } else {
                var
            })
        }
    }
}

impl CovState {
    pub fn weight(&self) -> f64 {
        self.weight
    }

    fn new(x: &[f64], y: &[f64]) -> Self {
        assert!(x.len() == y.len());
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let mean_x = anchored_mean(x, weight);
        let mean_y = anchored_mean(y, weight);
        Self {
            weight,
            mean_x,
            mean_y,
            dp_xy: alg_sum_f64(
                x.iter()
                    .zip(y)
                    .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)),
            ),
        }
    }

    pub fn insert_one(&mut self, x: f64, y: f64) {
        let new_weight = self.weight + 1.0;
        let new_weight_frac = 1.0 / new_weight;
        let delta_mean_x = x - self.mean_x;
        let delta_mean_y = y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * new_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * new_weight_frac;
        self.dp_xy += (x - new_mean_x) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean_x = other.mean_x - self.mean_x;
        let delta_mean_y = other.mean_y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * other_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * other_weight_frac;
        self.dp_xy += other.dp_xy + other.weight * (other.mean_x - new_mean_x) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn finalize(&self, ddof: u8) -> Option<f64> {
        if self.weight <= ddof as f64 {
            None
        } else {
            Some(self.dp_xy / (self.weight - ddof as f64))
        }
    }
}

impl PearsonState {
    pub fn weight(&self) -> f64 {
        self.weight
    }

    fn new(x: &[f64], y: &[f64]) -> Self {
        assert!(x.len() == y.len());
        if x.is_empty() {
            return Self::default();
        }

        let weight = x.len() as f64;
        let mean_x = anchored_mean(x, weight);
        let mean_y = anchored_mean(y, weight);
        let mut dp_xx = 0.0;
        let mut dp_xy = 0.0;
        let mut dp_yy = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            dp_xx = alg_add_f64(dp_xx, (xi - mean_x) * (xi - mean_x));
            dp_xy = alg_add_f64(dp_xy, (xi - mean_x) * (yi - mean_y));
            dp_yy = alg_add_f64(dp_yy, (yi - mean_y) * (yi - mean_y));
        }
        Self {
            weight,
            mean_x,
            mean_y,
            dp_xx,
            dp_xy,
            dp_yy,
        }
    }

    pub fn insert_one(&mut self, x: f64, y: f64) {
        let new_weight = self.weight + 1.0;
        let new_weight_frac = 1.0 / new_weight;
        let delta_mean_x = x - self.mean_x;
        let delta_mean_y = y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * new_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * new_weight_frac;
        self.dp_xx += (x - new_mean_x) * delta_mean_x;
        self.dp_xy += (x - new_mean_x) * delta_mean_y;
        self.dp_yy += (y - new_mean_y) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let other_weight_frac = other.weight / new_weight;
        let delta_mean_x = other.mean_x - self.mean_x;
        let delta_mean_y = other.mean_y - self.mean_y;
        let new_mean_x = self.mean_x + delta_mean_x * other_weight_frac;
        let new_mean_y = self.mean_y + delta_mean_y * other_weight_frac;
        self.dp_xx += other.dp_xx + other.weight * (other.mean_x - new_mean_x) * delta_mean_x;
        self.dp_xy += other.dp_xy + other.weight * (other.mean_x - new_mean_x) * delta_mean_y;
        self.dp_yy += other.dp_yy + other.weight * (other.mean_y - new_mean_y) * delta_mean_y;
        self.weight = new_weight;
        self.mean_x = new_mean_x;
        self.mean_y = new_mean_y;
    }

    pub fn finalize(&self) -> f64 {
        let denom_sq = self.dp_xx * self.dp_yy;
        if denom_sq > 0.0 {
            self.dp_xy / denom_sq.sqrt()
        } else {
            f64::NAN
        }
    }
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct SkewState {
    weight: f64,
    mean: f64,
    m2: f64,
    m3: f64,
}

impl SkewState {
    fn new(x: &[f64]) -> Self {
        Self::from_iter(x.iter().copied(), x.len())
    }

    fn from_iter(iter: impl Iterator<Item = f64> + Clone, length: usize) -> Self {
        if length == 0 {
            return Self::default();
        }

        let weight = length as f64;
        let mean = alg_sum_f64(iter.clone()) / weight;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        for xi in iter {
            let d = xi - mean;
            let d2 = d * d;
            let d3 = d * d2;
            m2 = alg_add_f64(m2, d2);
            m3 = alg_add_f64(m3, d3);
        }
        Self {
            weight,
            mean,
            m2,
            m3,
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            self.m3 = 0.0;
        }
    }

    pub fn from_array(arr: &PrimitiveArray<f64>, start: usize, length: usize) -> Self {
        let validity = arr.validity().cloned();
        let validity = validity
            .map(|v| v.sliced(start, length))
            .filter(|v| v.unset_bits() > 0);

        match validity {
            None => Self::new(&arr.values().as_slice()[start..][..length]),
            Some(validity) => {
                let iter = arr.values()[start..][..length].iter().copied();
                let iter = iter
                    .zip(validity.iter())
                    .filter_map(|(x, v)| v.then_some(x));
                Self::from_iter(iter, validity.set_bits())
            },
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Specialization of self.combine(&SkewState { weight: 1.0, mean: x, m2: 0.0, m3: 0.0 });
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + delta_mean_weight;

        let weight_diff = self.weight - 1.0;
        let m2_update = (x - new_mean) * delta_mean;
        let new_m2 = self.m2 + m2_update;
        let new_m3 = self.m3 + delta_mean_weight * (m2_update * weight_diff - 3.0 * self.m2);

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let delta_mean = other.mean - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + other.weight * delta_mean_weight;

        let weight_diff = self.weight - other.weight;
        let self_weight_other_m2 = self.weight * other.m2;
        let other_weight_self_m2 = other.weight * self.m2;
        let m2_update = other.weight * (other.mean - new_mean) * delta_mean;
        let new_m2 = self.m2 + other.m2 + m2_update;
        let new_m3 = self.m3
            + other.m3
            + delta_mean_weight
                * (m2_update * weight_diff + 3.0 * (self_weight_other_m2 - other_weight_self_m2));

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, bias: bool) -> Option<f64> {
        let m2 = self.m2 / self.weight;
        let m3 = self.m3 / self.weight;
        let is_zero = m2 <= (f64::EPSILON * self.mean).powi(2);
        let biased_est = if is_zero { f64::NAN } else { m3 / m2.powf(1.5) };
        if bias {
            if self.weight == 0.0 {
                None
            } else {
                Some(biased_est)
            }
        } else {
            if self.weight <= 2.0 {
                None
            } else {
                let correction = (self.weight * (self.weight - 1.0)).sqrt() / (self.weight - 2.0);
                Some(correction * biased_est)
            }
        }
    }
}

#[derive(Default, Clone)]
#[repr(C)] // For serialization, don't change struct member order.
pub struct KurtosisState {
    weight: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl KurtosisState {
    pub fn new(x: &[f64]) -> Self {
        Self::from_iter(x.iter().copied(), x.len())
    }

    pub fn from_iter(iter: impl Iterator<Item = f64> + Clone, length: usize) -> Self {
        if length == 0 {
            return Self::default();
        }

        let weight = length as f64;
        let mean = alg_sum_f64(iter.clone()) / weight;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for xi in iter {
            let d = xi - mean;
            let d2 = d * d;
            let d3 = d * d2;
            let d4 = d2 * d2;
            m2 = alg_add_f64(m2, d2);
            m3 = alg_add_f64(m3, d3);
            m4 = alg_add_f64(m4, d4);
        }
        Self {
            weight,
            mean,
            m2,
            m3,
            m4,
        }
    }

    pub fn from_array(arr: &PrimitiveArray<f64>, start: usize, length: usize) -> Self {
        let validity = arr.validity().cloned();
        let validity = validity
            .map(|v| v.sliced(start, length))
            .filter(|v| v.unset_bits() > 0);

        match validity {
            None => Self::new(&arr.values().as_slice()[start..][..length]),
            Some(validity) => {
                let iter = arr.values()[start..][..length].iter().copied();
                let iter = iter
                    .zip(validity.iter())
                    .filter_map(|(x, v)| v.then_some(x));
                Self::from_iter(iter, validity.set_bits())
            },
        }
    }

    fn clear_zero_weight_nan(&mut self) {
        // Clear NaNs due to division by zero.
        if self.weight == 0.0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            self.m3 = 0.0;
            self.m4 = 0.0;
        }
    }

    pub fn insert_one(&mut self, x: f64) {
        // Specialization of self.combine(&KurtosisState { weight: 1.0, mean: x, m2: 0.0, m3: 0.0, m4: 0.0 });
        let new_weight = self.weight + 1.0;
        let delta_mean = x - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + delta_mean_weight;

        let weight_diff = self.weight - 1.0;
        let m2_update = (x - new_mean) * delta_mean;
        let new_m2 = self.m2 + m2_update;
        let new_m3 = self.m3 + delta_mean_weight * (m2_update * weight_diff - 3.0 * self.m2);
        let new_m4 = self.m4
            + delta_mean_weight
                * (delta_mean_weight
                    * (m2_update * (self.weight * weight_diff + 1.0) + 6.0 * self.m2)
                    - 4.0 * self.m3);

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        self.clear_zero_weight_nan();
    }

    pub fn combine(&mut self, other: &Self) {
        if other.weight == 0.0 {
            return;
        } else if self.weight == 0.0 {
            *self = other.clone();
            return;
        }

        let new_weight = self.weight + other.weight;
        let delta_mean = other.mean - self.mean;
        let delta_mean_weight = delta_mean / new_weight;
        let new_mean = self.mean + other.weight * delta_mean_weight;

        let weight_diff = self.weight - other.weight;
        let self_weight_other_m2 = self.weight * other.m2;
        let other_weight_self_m2 = other.weight * self.m2;
        let m2_update = other.weight * (other.mean - new_mean) * delta_mean;
        let new_m2 = self.m2 + other.m2 + m2_update;
        let new_m3 = self.m3
            + other.m3
            + delta_mean_weight
                * (m2_update * weight_diff + 3.0 * (self_weight_other_m2 - other_weight_self_m2));
        let new_m4 = self.m4
            + other.m4
            + delta_mean_weight
                * (delta_mean_weight
                    * (m2_update * (self.weight * weight_diff + other.weight * other.weight)
                        + 6.0
                            * (self.weight * self_weight_other_m2
                                + other.weight * other_weight_self_m2))
                    + 4.0 * (self.weight * other.m3 - other.weight * self.m3));

        self.weight = new_weight;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
        self.clear_zero_weight_nan();
    }

    pub fn finalize(&self, fisher: bool, bias: bool) -> Option<f64> {
        let m4 = self.m4 / self.weight;
        let m2 = self.m2 / self.weight;
        let is_zero = m2 <= (f64::EPSILON * self.mean).powi(2);
        let biased_est = if is_zero { f64::NAN } else { m4 / (m2 * m2) };
        let out = if bias {
            if self.weight == 0.0 {
                return None;
            }

            biased_est
        } else {
            if self.weight <= 3.0 {
                return None;
            }

            let n = self.weight;
            let nm1_nm2 = (n - 1.0) / (n - 2.0);
            let np1_nm3 = (n + 1.0) / (n - 3.0);
            let nm1_nm3 = (n - 1.0) / (n - 3.0);
            nm1_nm2 * (np1_nm3 * biased_est - 3.0 * nm1_nm3) + 3.0
        };

        if fisher { Some(out - 3.0) } else { Some(out) }
    }
}

fn chunk_as_float<T, I, F>(it: I, mut f: F)
where
    T: NativeType + AsPrimitive<f64>,
    I: IntoIterator<Item = T>,
    F: FnMut(&[f64]),
{
    let mut chunk = [0.0; CHUNK_SIZE];
    let mut i = 0;
    for val in it {
        if i >= CHUNK_SIZE {
            f(&chunk);
            i = 0;
        }
        chunk[i] = val.as_();
        i += 1;
    }
    if i > 0 {
        f(&chunk[..i]);
    }
}

fn chunk_as_float_binary<T, U, I, F>(it: I, mut f: F)
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
    I: IntoIterator<Item = (T, U)>,
    F: FnMut(&[f64], &[f64]),
{
    let mut left_chunk = [0.0; CHUNK_SIZE];
    let mut right_chunk = [0.0; CHUNK_SIZE];
    let mut i = 0;
    for (l, r) in it {
        if i >= CHUNK_SIZE {
            f(&left_chunk, &right_chunk);
            i = 0;
        }
        left_chunk[i] = l.as_();
        right_chunk[i] = r.as_();
        i += 1;
    }
    if i > 0 {
        f(&left_chunk[..i], &right_chunk[..i]);
    }
}

pub fn var<T>(arr: &PrimitiveArray<T>) -> VarState
where
    T: NativeType + AsPrimitive<f64>,
{
    let mut out = VarState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.non_null_values_iter(), |chunk| {
            out.combine(&VarState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values().iter().copied(), |chunk| {
            out.combine(&VarState::new(chunk))
        });
    }
    out
}

pub fn cov<T, U>(x: &PrimitiveArray<T>, y: &PrimitiveArray<U>) -> CovState
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
{
    assert!(x.len() == y.len());
    let mut out = CovState::default();
    if x.has_nulls() || y.has_nulls() {
        chunk_as_float_binary(
            x.iter()
                .zip(y.iter())
                .filter_map(|(l, r)| l.copied().zip(r.copied())),
            |l, r| out.combine(&CovState::new(l, r)),
        );
    } else {
        chunk_as_float_binary(
            x.values().iter().copied().zip(y.values().iter().copied()),
            |l, r| out.combine(&CovState::new(l, r)),
        );
    }
    out
}

pub fn pearson_corr<T, U>(x: &PrimitiveArray<T>, y: &PrimitiveArray<U>) -> PearsonState
where
    T: NativeType + AsPrimitive<f64>,
    U: NativeType + AsPrimitive<f64>,
{
    assert!(x.len() == y.len());
    let mut out = PearsonState::default();
    if x.has_nulls() || y.has_nulls() {
        chunk_as_float_binary(
            x.iter()
                .zip(y.iter())
                .filter_map(|(l, r)| l.copied().zip(r.copied())),
            |l, r| out.combine(&PearsonState::new(l, r)),
        );
    } else {
        chunk_as_float_binary(
            x.values().iter().copied().zip(y.values().iter().copied()),
            |l, r| out.combine(&PearsonState::new(l, r)),
        );
    }
    out
}

pub fn skew<T>(arr: &PrimitiveArray<T>) -> SkewState
where
    T: NativeType + AsPrimitive<f64>,
{
    let mut out = SkewState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.non_null_values_iter(), |chunk| {
            out.combine(&SkewState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values().iter().copied(), |chunk| {
            out.combine(&SkewState::new(chunk))
        });
    }
    out
}

pub fn kurtosis<T>(arr: &PrimitiveArray<T>) -> KurtosisState
where
    T: NativeType + AsPrimitive<f64>,
{
    let mut out = KurtosisState::default();
    if arr.has_nulls() {
        chunk_as_float(arr.non_null_values_iter(), |chunk| {
            out.combine(&KurtosisState::new(chunk))
        });
    } else {
        chunk_as_float(arr.values().iter().copied(), |chunk| {
            out.combine(&KurtosisState::new(chunk))
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant column must be *exactly* degenerate, no matter the constant.
    const CONSTANTS: [f64; 7] = [0.0, 0.01, 0.3, 0.7, 1.0, 1.0e9, -9.570205894681823];

    /// Lengths that don't line up with `CHUNK_SIZE`, plus very small and very
    /// large ones.
    const LENGTHS: [usize; 4] = [5, 117, 245, 2097];

    /// Neumaier (compensated) summation, used as a high accuracy reference.
    fn neumaier_sum(it: impl IntoIterator<Item = f64>) -> f64 {
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;
        for value in it {
            let new_sum = sum + value;
            if sum.abs() >= value.abs() {
                compensation += (sum - new_sum) + value;
            } else {
                compensation += (value - new_sum) + sum;
            }
            sum = new_sum;
        }
        sum + compensation
    }

    fn reference_mean(x: &[f64]) -> f64 {
        neumaier_sum(x.iter().copied()) / x.len() as f64
    }

    fn reference_var(x: &[f64], ddof: f64) -> f64 {
        let mean = reference_mean(x);
        let dp = neumaier_sum(x.iter().map(|&xi| (xi - mean) * (xi - mean)));
        dp / (x.len() as f64 - ddof)
    }

    fn reference_cov(x: &[f64], y: &[f64], ddof: f64) -> f64 {
        let mean_x = reference_mean(x);
        let mean_y = reference_mean(y);
        let dp = neumaier_sum(
            x.iter()
                .zip(y)
                .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)),
        );
        dp / (x.len() as f64 - ddof)
    }

    /// Deterministic pseudo-random values in `[-1, 1)`, so the tests don't need
    /// an RNG dependency.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_rel_eq(actual: f64, expected: f64, rtol: f64, msg: &str) {
        let scale = expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= rtol * scale,
            "{msg}: actual={actual}, expected={expected}, rtol={rtol}"
        );
    }

    /// A constant column has no variance at all: `var == 0`, `cov == 0` and
    /// `corr == NaN`, for every constant, every length and both ddofs.
    #[test]
    fn test_constant_column_is_exactly_degenerate() {
        for c in CONSTANTS {
            for n in LENGTHS {
                let x = vec![c; n];
                // Two `y` shapes, so the result can't accidentally depend on the
                // values of the other column.
                let y_linear: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let y_periodic: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();

                let x_arr = PrimitiveArray::<f64>::from_slice(&x);

                for ddof in [0, 1] {
                    let got = var(&x_arr).finalize(ddof).unwrap();
                    assert_eq!(got, 0.0, "var(c={c}, n={n}, ddof={ddof}) = {got}");
                }

                for y in [&y_linear, &y_periodic] {
                    let y_arr = PrimitiveArray::<f64>::from_slice(y);

                    for ddof in [0, 1] {
                        let got = cov(&x_arr, &y_arr).finalize(ddof).unwrap();
                        assert_eq!(got, 0.0, "cov(c={c}, n={n}, ddof={ddof}) = {got}");
                    }

                    let got = pearson_corr(&x_arr, &y_arr).finalize();
                    assert!(
                        got.is_nan(),
                        "corr(c={c}, n={n}, len(y)={}) = {got}",
                        y.len()
                    );
                }
            }
        }
    }

    /// Chunking must not change the result: a chunk of a constant column has
    /// exactly zero spread, so `combine()` keeps `dp == 0`.
    #[test]
    fn test_constant_column_across_chunks() {
        for c in CONSTANTS {
            for n in LENGTHS {
                let x = vec![c; n];
                let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();

                let mut layouts = vec![1, 2, 3, 7, 63, 127, 128, 129, 1024, n];
                layouts.retain(|&chunk_size| chunk_size <= n);
                layouts.dedup();

                let mut results = Vec::new();
                for &chunk_size in &layouts {
                    let mut var_state = VarState::default();
                    let mut cov_state = CovState::default();
                    let mut corr_state = PearsonState::default();

                    for (xc, yc) in x.chunks(chunk_size).zip(y.chunks(chunk_size)) {
                        // The invariant that makes `combine()` exact.
                        let chunk_var = VarState::new(xc);
                        assert_eq!(chunk_var.mean, c, "mean(c={c}, chunk={chunk_size})");
                        assert_eq!(chunk_var.dp, 0.0, "dp(c={c}, chunk={chunk_size})");

                        var_state.combine(&chunk_var);
                        cov_state.combine(&CovState::new(xc, yc));
                        corr_state.combine(&PearsonState::new(xc, yc));
                    }

                    let var = var_state.finalize(0).unwrap();
                    let cov = cov_state.finalize(0).unwrap();
                    let corr = corr_state.finalize();
                    assert_eq!(var, 0.0, "var(c={c}, n={n}, chunk={chunk_size}) = {var}");
                    assert_eq!(cov, 0.0, "cov(c={c}, n={n}, chunk={chunk_size}) = {cov}");
                    assert!(
                        corr.is_nan(),
                        "corr(c={c}, n={n}, chunk={chunk_size}) = {corr}"
                    );
                    results.push((var, cov, format!("{corr:?}")));
                }

                for result in results.iter().skip(1) {
                    assert_eq!(
                        result, &results[0],
                        "chunk layout changed the result (c={c}, n={n})"
                    );
                }
            }
        }
    }

    /// Non-degenerate inputs must keep their accuracy: compare against a
    /// compensated-summation reference, and check the analytic `corr == ±1`
    /// case for exactly affine related columns.
    #[test]
    fn test_non_degenerate_not_regressed() {
        for (n, seed) in [(64usize, 1u64), (245, 2), (1000, 3), (4097, 4)] {
            let x = pseudo_random(n, seed);
            let y = pseudo_random(n, seed.wrapping_mul(7919));
            let x_arr = PrimitiveArray::<f64>::from_slice(&x);
            let y_arr = PrimitiveArray::<f64>::from_slice(&y);

            let expected_var_x = reference_var(&x, 1.0);
            let expected_var_y = reference_var(&y, 1.0);
            let expected_cov = reference_cov(&x, &y, 1.0);

            assert_rel_eq(
                var(&x_arr).finalize(1).unwrap(),
                expected_var_x,
                1e-12,
                &format!("var(n={n})"),
            );
            assert_rel_eq(
                cov(&x_arr, &y_arr).finalize(1).unwrap(),
                expected_cov,
                1e-12,
                &format!("cov(n={n})"),
            );

            let corr = pearson_corr(&x_arr, &y_arr).finalize();
            let expected_corr = expected_cov / (expected_var_x.sqrt() * expected_var_y.sqrt());
            assert_rel_eq(corr, expected_corr, 1e-12, &format!("corr(n={n})"));
            assert!(
                corr.is_finite() && corr.abs() <= 1.0,
                "corr(n={n}) = {corr}"
            );
        }

        // An exact affine relation must give exactly ±1, for large inputs too.
        let x: Vec<f64> = (0..2097).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
        let y_neg: Vec<f64> = x.iter().map(|&xi| -xi + 5.0).collect();
        let x_arr = PrimitiveArray::<f64>::from_slice(&x);
        assert_rel_eq(
            pearson_corr(&x_arr, &PrimitiveArray::<f64>::from_slice(&y)).finalize(),
            1.0,
            1e-15,
            "corr(x, 2x + 3)",
        );
        assert_rel_eq(
            pearson_corr(&x_arr, &PrimitiveArray::<f64>::from_slice(&y_neg)).finalize(),
            -1.0,
            1e-15,
            "corr(x, -x + 5)",
        );
    }

    /// Chunked and unchunked results must agree for non-degenerate data as well.
    #[test]
    fn test_chunking_stability_non_degenerate() {
        let n = 2100;
        let x = pseudo_random(n, 42);
        let y = pseudo_random(n, 1234);

        let single = PearsonState::new(&x, &y).finalize();

        for chunk_size in [128usize, 255, 512, 1024] {
            let mut state = PearsonState::default();
            for (xc, yc) in x.chunks(chunk_size).zip(y.chunks(chunk_size)) {
                state.combine(&PearsonState::new(xc, yc));
            }
            assert_rel_eq(
                state.finalize(),
                single,
                1e-12,
                &format!("corr chunk_size={chunk_size}"),
            );
        }
    }
}
