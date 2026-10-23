//! Opt-in batched precision rescue with an independent convergence check.
//!
//! The ordinary workspace runs first. Selected samples are then evaluated as
//! one batch at each of two higher precisions, including scalar masters. Every
//! Laurent coefficient and Taylor component must converge before the result
//! is rounded back to the caller's numeric domain. This is a numerical
//! convergence check, not a certified error bound.

use iregi_masters::NativeFloat;
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        float::{Complex, DoubleFloat, Float, FloatLike, Real, RealLike, SingleFloat},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

use crate::{
    execution::{ExecutionStats, Workspace},
    plan::ReductionPlan,
    types::{BatchResult, KinematicsBatch, LaneStatus, ReductionError},
};

/// Which otherwise valid samples require a higher-precision check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RescueTrigger {
    /// Retry samples rejected by the ordinary workspace's stability monitor.
    #[default]
    UnstableOnly,
    /// Also verify successful samples. Useful when qualifying new families:
    /// local cancellation estimates do not bound accumulated rounding error.
    VerifyAllSuccessful,
}

/// Working precisions for rescue and its separate verification evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RescueSettings {
    pub working_bits: u32,
    pub verification_bits: u32,
    /// Requested accuracy uses the workspace convention `2^(19-bits)`.
    /// `None` retains the precision of the caller's output domain.
    pub accuracy_bits: Option<u32>,
    pub trigger: RescueTrigger,
}

impl Default for RescueSettings {
    fn default() -> Self {
        Self {
            working_bits: 128,
            verification_bits: 256,
            accuracy_bits: None,
            trigger: RescueTrigger::UnstableOnly,
        }
    }
}

/// Per-sample precision diagnostics, separate from reduction-method fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RescueStatus {
    NotRequested,
    Converged {
        working_bits: u32,
        verification_bits: u32,
        /// `None` means the primary reduction failed; `Some(false)` exposes a
        /// successful primary result that missed the requested accuracy.
        primary_agreed: Option<bool>,
    },
    Rejected(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RescueStats {
    pub primary: ExecutionStats,
    pub working: ExecutionStats,
    pub verification: ExecutionStats,
    pub requested_rows: usize,
    pub accepted_rows: usize,
    pub corrected_successful_rows: usize,
}

/// Numeric conversion required by precision rescue.
///
/// Promotion must preserve the exact represented input, including the sign of
/// zero. Implementations reject a configured precision that cannot do so.
/// Rounding is used only after convergence, at the public output boundary.
pub trait RescueScalar: NativeFloat {
    fn promote_exact(&self, bits: u32) -> Result<Float, String>;
    fn round_rescued(&self, value: &Float) -> Self;
}

impl RescueScalar for f64 {
    fn promote_exact(&self, bits: u32) -> Result<Float, String> {
        let promoted = Float::with_val(bits, *self);
        if promoted.to_f64() == *self && (bits >= 53 || !self.is_finite()) {
            Ok(promoted)
        } else {
            Err("rescue precision cannot preserve the binary64 input".into())
        }
    }
    fn round_rescued(&self, value: &Float) -> Self {
        value.to_f64()
    }
}

impl RescueScalar for DoubleFloat {
    fn promote_exact(&self, bits: u32) -> Result<Float, String> {
        let pair = self.into_inner();
        let hi = Float::with_val(53, pair.hi());
        let lo = Float::with_val(53, pair.lo());
        let exact = hi.try_to_rational().ok_or("nonfinite DoubleFloat input")?
            + lo.try_to_rational().ok_or("nonfinite DoubleFloat input")?;
        let promoted = if exact.is_zero() {
            Float::with_val(bits, if self.negative_sign() { -0.0 } else { 0.0 })
        } else {
            Float::new(bits).from_rational(&exact)
        };
        if promoted.try_to_rational().as_ref() == Some(&exact) {
            Ok(promoted)
        } else {
            Err(
                "rescue precision cannot preserve both DoubleFloat limbs; increase working_bits"
                    .into(),
            )
        }
    }
    fn round_rescued(&self, value: &Float) -> Self {
        // The upstream compensated-sum conversion combines -0 with a +0
        // residual. Retain the sign explicitly for branch-sensitive callers.
        if value.is_zero() {
            DoubleFloat::from(if value.is_sign_negative() { -0.0 } else { 0.0 })
        } else {
            value.to_double_float()
        }
    }
}

impl RescueScalar for Float {
    fn promote_exact(&self, bits: u32) -> Result<Float, String> {
        let promoted = self.at_precision(bits);
        if &promoted == self {
            Ok(promoted)
        } else {
            Err("rescue precision cannot preserve the arbitrary-precision input; increase working_bits".into())
        }
    }
    fn round_rescued(&self, value: &Float) -> Self {
        value.at_precision(self.get_precision())
    }
}

/// Reusable primary and higher-precision workspaces for one reduction plan.
///
/// Invalid inputs and scalar-master domain failures remain failures. Rescue
/// does not alter kinematics, propagator prescriptions, or Dual branch rules.
/// All selected lanes run together at each precision; derivatives are still
/// Taylor components within a lane. Structural preparation is paid once.
#[derive(Clone)]
pub struct PrecisionRescueWorkspace<T: RescueScalar> {
    prototype: T,
    plan: ReductionPlan,
    primary: Workspace<T>,
    working: Workspace<Float>,
    verification: Workspace<Float>,
    settings: RescueSettings,
    accuracy_bits: u32,
    diagnostics: Vec<RescueStatus>,
    primary_statuses: Vec<LaneStatus>,
    stats: RescueStats,
}

impl<T: RescueScalar> PrecisionRescueWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(
        plan: &ReductionPlan,
        prototype: T,
        settings: RescueSettings,
    ) -> Result<Self, ReductionError> {
        Self::prepare::<HyperDual<Complex<Rational>>>(plan, prototype, settings, None)
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        settings: RescueSettings,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, settings, Some(&dual))
    }

    fn prepare<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        settings: RescueSettings,
        dual: Option<&D>,
    ) -> Result<Self, ReductionError> {
        let output_bits = prototype.get_precision();
        let accuracy_bits = settings.accuracy_bits.unwrap_or(output_bits);
        if accuracy_bits == 0
            || accuracy_bits > output_bits
            || settings.working_bits <= output_bits
            || settings.verification_bits <= settings.working_bits
        {
            return Err(ReductionError::InvalidInput(
                "rescue requires 0 < accuracy_bits <= output bits < working_bits < verification_bits".into(),
            ));
        }
        // Specialize the mathematical plan once for this Dual algebra. Calling
        // Workspace::new_dual with the original zero-jet plan at each precision
        // would otherwise repeat the complete symbolic tensor preparation.
        let specialized = if let Some(dual) = dual {
            iregi_masters::validate_dual_structure(dual)
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
            let order = u16::try_from(dual.get_max_depth()).map_err(|_| {
                ReductionError::Preparation("Dual jet order exceeds the supported u16 range".into())
            })?;
            if order == plan.analytic_jet_order {
                None
            } else {
                Some(plan.with_analytic_jet_order(order)?)
            }
        } else {
            None
        };
        let plan = specialized.as_deref().unwrap_or(plan);
        #[cfg(test)]
        let started = std::time::Instant::now();
        let mut primary = match dual {
            Some(dual) => Workspace::<T>::new_dual(plan, prototype.clone(), BorrowedDual(dual))?,
            None => Workspace::<T>::new(plan, prototype.clone())?,
        };
        #[cfg(test)]
        preparation_marker("primary workspace prepared", started);
        let mut working = match dual {
            Some(dual) => Workspace::<Float>::new_dual(
                plan,
                Float::new(settings.working_bits),
                BorrowedDual(dual),
            )?,
            None => Workspace::<Float>::new(plan, Float::new(settings.working_bits))?,
        };
        #[cfg(test)]
        preparation_marker("working-precision workspace prepared", started);
        let mut verification = match dual {
            Some(dual) => Workspace::<Float>::new_dual(
                plan,
                Float::new(settings.verification_bits),
                BorrowedDual(dual),
            )?,
            None => Workspace::<Float>::new(plan, Float::new(settings.verification_bits))?,
        };
        #[cfg(test)]
        preparation_marker("verification workspace prepared", started);
        primary.set_accuracy_precision(accuracy_bits)?;
        working.set_accuracy_precision(accuracy_bits)?;
        verification.set_accuracy_precision(accuracy_bits)?;
        Ok(Self {
            prototype,
            plan: plan.clone(),
            primary,
            working,
            verification,
            settings,
            accuracy_bits,
            diagnostics: Vec::new(),
            primary_statuses: Vec::new(),
            stats: RescueStats::default(),
        })
    }

    pub fn last_execution(&self) -> RescueStats {
        self.stats
    }

    pub fn last_rescue_statuses(&self) -> &[RescueStatus] {
        &self.diagnostics
    }

    /// Status before a precision retry, including successful samples that were
    /// selected for independent accuracy verification.
    pub fn last_primary_statuses(&self) -> &[LaneStatus] {
        &self.primary_statuses
    }

    pub fn component_count(&self) -> usize {
        self.primary.component_count()
    }

    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.primary.dual_shape()
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.stats = RescueStats::default();
        self.diagnostics.clear();
        self.primary_statuses.clear();
        let mut result = self.primary.reduce_batch(batch)?;
        self.stats.primary = self.primary.last_execution();
        self.primary_statuses.clone_from(&result.statuses);
        self.diagnostics
            .resize(batch.rows, RescueStatus::NotRequested);
        let input_width = self.plan.parameters.len() * batch.components;
        let output_width = result.basis.len() * 4 * batch.components;
        let mut lanes = Vec::new();
        let mut low_inputs = Vec::new();
        let mut high_inputs = Vec::new();
        for lane in 0..batch.rows {
            let requested = match result.statuses[lane] {
                LaneStatus::Unstable(_) => true,
                LaneStatus::Success => self.settings.trigger == RescueTrigger::VerifyAllSuccessful,
                _ => false,
            };
            if !requested {
                continue;
            }
            self.stats.requested_rows += 1;
            let row = &batch.values[lane * input_width..(lane + 1) * input_width];
            match promote_row(
                row,
                self.settings.working_bits,
                self.settings.verification_bits,
            ) {
                Ok((low, high)) => {
                    lanes.push(lane);
                    low_inputs.extend(low);
                    high_inputs.extend(high);
                }
                Err(reason) => reject(&mut result, &mut self.diagnostics, lane, reason),
            }
        }
        if lanes.is_empty() {
            return Ok(result);
        }
        let low_batch = KinematicsBatch {
            rows: lanes.len(),
            components: batch.components,
            values: low_inputs,
        };
        let high_batch = KinematicsBatch {
            rows: lanes.len(),
            components: batch.components,
            values: high_inputs,
        };
        let low = self.working.reduce_batch(&low_batch)?;
        let high = self.verification.reduce_batch(&high_batch)?;
        self.stats.working = self.working.last_execution();
        self.stats.verification = self.verification.last_execution();
        for (compact, &lane) in lanes.iter().enumerate() {
            let attempt = (|| {
                let low = low.lane(compact).map_err(|status| {
                    format!(
                        "{}-bit reduction returned {status:?}",
                        self.settings.working_bits,
                    )
                })?;
                let high = high.lane(compact).map_err(|status| {
                    format!(
                        "{}-bit verification returned {status:?}",
                        self.settings.verification_bits,
                    )
                })?;
                let input = &high_batch.values[compact * input_width..(compact + 1) * input_width];
                self.compare(low, high, input, batch.components, 8)?;
                #[cfg(test)]
                if std::env::var_os("IREGI_RESCUE_PROFILE").is_some()
                    && let Some(moment) = self
                        .plan
                        .basis
                        .iter()
                        .position(|moment| moment.0 == [5, 0, 0, 1])
                {
                    let index = (moment * 4 + 3) * batch.components;
                    eprintln!(
                        "precision coefficient lane={lane}, moment=[5,0,0,1], slot=finite, Taylor component=0: bits={} real={} imag={}; bits={} real={} imag={}",
                        self.settings.working_bits,
                        low[index].re,
                        low[index].im,
                        self.settings.verification_bits,
                        high[index].re,
                        high[index].im,
                    );
                }
                let rounded: Vec<_> = high
                    .iter()
                    .map(|value| {
                        Complex::new(
                            self.prototype.round_rescued(&value.re),
                            self.prototype.round_rescued(&value.im),
                        )
                    })
                    .collect();
                if rounded
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(
                        "converged result is outside the output domain's finite range".into(),
                    );
                }
                let promoted: Result<Vec<_>, String> = rounded
                    .iter()
                    .map(|value| {
                        Ok(Complex::new(
                            value.re.promote_exact(self.settings.verification_bits)?,
                            value.im.promote_exact(self.settings.verification_bits)?,
                        ))
                    })
                    .collect();
                self.compare(&promoted?, high, input, batch.components, 0)?;
                let primary_agreed = result.lane(lane).ok().map(|primary| {
                    let promoted: Result<Vec<_>, String> = primary
                        .iter()
                        .map(|value| {
                            Ok(Complex::new(
                                value.re.promote_exact(self.settings.verification_bits)?,
                                value.im.promote_exact(self.settings.verification_bits)?,
                            ))
                        })
                        .collect();
                    promoted
                        .and_then(|values| self.compare(&values, high, input, batch.components, 0))
                        .is_ok()
                });
                Ok((rounded, primary_agreed))
            })();
            match attempt {
                Ok((values, primary_agreed)) => {
                    result.values[lane * output_width..(lane + 1) * output_width]
                        .clone_from_slice(&values);
                    result.statuses[lane] = LaneStatus::Success;
                    result.fallback_used[lane] |=
                        low.fallback_used[compact] || high.fallback_used[compact];
                    result.analytic_used[lane] |=
                        low.analytic_used[compact] || high.analytic_used[compact];
                    self.diagnostics[lane] = RescueStatus::Converged {
                        working_bits: self.settings.working_bits,
                        verification_bits: self.settings.verification_bits,
                        primary_agreed,
                    };
                    self.stats.accepted_rows += 1;
                    if primary_agreed == Some(false) {
                        self.stats.corrected_successful_rows += 1;
                    }
                }
                Err(reason) => reject(&mut result, &mut self.diagnostics, lane, reason),
            }
        }
        Ok(result)
    }

    /// Flatten static or dynamic Symbolica Dual values, preserving their exact
    /// Taylor layout rather than treating components as additional samples.
    pub fn reduce_typed_dual_batch<D: DualNumber<Complex<T>>>(
        &mut self,
        input: &[D],
        rows: usize,
    ) -> Result<BatchResult<T>, ReductionError> {
        let shape = self.dual_shape().ok_or_else(|| {
            ReductionError::InvalidInput(
                "use PrecisionRescueWorkspace::new_dual before passing Dual values".into(),
            )
        })?;
        let expected = rows
            .checked_mul(self.plan.parameters.len())
            .ok_or_else(|| {
                ReductionError::InvalidInput("rescue input dimensions overflow".into())
            })?;
        if input.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "expected {expected} Dual input values, received {}",
                input.len(),
            )));
        }
        let expected_shape: Vec<_> = shape.iter().map(Vec::as_slice).collect();
        let components = self.component_count();
        let mut values = Vec::new();
        for value in input {
            if value.get_shape() != expected_shape || value.values().len() != components {
                return Err(ReductionError::InvalidInput(
                    "input Dual shape differs from rescue workspace".into(),
                ));
            }
            values.extend_from_slice(value.values());
        }
        self.reduce_batch(&KinematicsBatch {
            rows,
            components,
            values,
        })
    }

    fn compare(
        &self,
        low: &[Complex<Float>],
        high: &[Complex<Float>],
        input: &[Complex<Float>],
        components: usize,
        guard_bits: u32,
    ) -> Result<(), String> {
        if low.len() != high.len() || high.len() != self.plan.basis.len() * 4 * components {
            return Err("inconsistent precision-verification output dimensions".into());
        }
        let bits = self.settings.verification_bits;
        let prototype = Float::new(bits);
        let mut scale = prototype.zero();
        for index in self.plan.layout.shifts.clone() {
            scale = max(scale, magnitude(&input[index * components]));
        }
        for index in self.plan.layout.masses.clone().chain(
            self.plan
                .layout
                .scalar_products
                .clone()
                .into_iter()
                .flatten(),
        ) {
            scale = max(scale, magnitude(&input[index * components]).sqrt());
        }
        if scale.is_zero() {
            scale = input[self.plan.layout.scale * components].re.clone();
        }
        let exponent = 19i64 - i64::from(self.accuracy_bits) - i64::from(guard_bits);
        let power = Float::with_val(bits, 2).pow(exponent.unsigned_abs());
        let tolerance = if exponent < 0 { power.inv() } else { power };
        let base_dimension = 4 + 2 * i32::from(self.plan.settings.dimension_shift)
            - 2 * self
                .plan
                .settings
                .powers
                .iter()
                .map(|&power| i32::from(power))
                .sum::<i32>();
        for (moment, monomial) in self.plan.basis.iter().enumerate() {
            let dimension = base_dimension + monomial.rank() as i32;
            let natural = scale.pow(u64::from(dimension.unsigned_abs()));
            let natural = if dimension < 0 {
                natural.inv()
            } else {
                natural
            };
            if !natural.is_finite() || natural.is_zero() {
                return Err(
                    "precision-verification normalization is outside the numeric range".into(),
                );
            }
            for slot in 0..4 {
                for component in 0..components {
                    let index = (moment * 4 + slot) * components + component;
                    let left = &low[index];
                    let right = &high[index];
                    let delta = Complex::new(
                        left.re.at_precision(bits) - &right.re,
                        left.im.at_precision(bits) - &right.im,
                    );
                    let error = magnitude(&delta);
                    let reference = max(natural.clone(), max(magnitude(left), magnitude(right)));
                    if !error.is_finite() || error > tolerance.clone() * reference {
                        return Err(format!(
                            "precision convergence failed for moment {:?}, Laurent slot {slot}, Taylor component {component}",
                            monomial.0,
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn promote_row<T: RescueScalar>(
    values: &[Complex<T>],
    low_bits: u32,
    high_bits: u32,
) -> Result<(Vec<Complex<Float>>, Vec<Complex<Float>>), String> {
    let mut low = Vec::with_capacity(values.len());
    let mut high = Vec::with_capacity(values.len());
    for (component, value) in values.iter().enumerate() {
        let re = value
            .re
            .promote_exact(low_bits)
            .map_err(|reason| format!("input component {component}, real part: {reason}",))?;
        let im = value
            .im
            .promote_exact(low_bits)
            .map_err(|reason| format!("input component {component}, imaginary part: {reason}",))?;
        high.push(Complex::new(
            re.at_precision(high_bits),
            im.at_precision(high_bits),
        ));
        low.push(Complex::new(re, im));
    }
    Ok((low, high))
}

fn reject<T: RescueScalar>(
    result: &mut BatchResult<T>,
    diagnostics: &mut [RescueStatus],
    lane: usize,
    reason: String,
) {
    result.statuses[lane] = LaneStatus::Unstable(format!("precision rescue rejected: {reason}"));
    diagnostics[lane] = RescueStatus::Rejected(reason);
}

fn magnitude(value: &Complex<Float>) -> Float {
    NativeFloat::hypot(&value.re, &value.im)
}

fn max(a: Float, b: Float) -> Float {
    if a > b { a } else { b }
}

#[cfg(test)]
fn preparation_marker(label: &str, started: std::time::Instant) {
    if std::env::var_os("IREGI_RESCUE_PROFILE").is_some() {
        eprintln!("{label} after {:?}", started.elapsed());
    }
}

struct BorrowedDual<'a, D>(&'a D);
impl<D: DualNumberStructure> DualNumberStructure for BorrowedDual<'_, D> {
    fn get_len(&self) -> usize {
        self.0.get_len()
    }
    fn get_shape(&self) -> Vec<&[usize]> {
        self.0.get_shape()
    }
    fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
        self.0.get_multiplication_table()
    }
    fn get_max_depth(&self) -> usize {
        self.0.get_max_depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, Convention, ReductionSettings};

    #[test]
    fn exact_promotions_preserve_low_limbs_and_signed_zero() {
        let tiny = 2.0_f64.powi(-1000);
        let sparse = DoubleFloat::from_compensated_sum(1.0, tiny);
        assert!(sparse.promote_exact(256).is_err());
        let full = sparse.promote_exact(1024).unwrap();
        assert_eq!(full - Float::with_val(1024, 1), Float::with_val(1024, tiny));
        let normal = DoubleFloat::from_compensated_sum(1.0, 2.0_f64.powi(-80));
        let exact = normal.promote_exact(128).unwrap();
        assert_ne!(exact, Float::with_val(128, 1));
        assert_eq!(normal.round_rescued(&exact), normal);
        let deep = Float::with_val(384, 1) + Float::with_val(384, 2).pow(300).inv();
        assert!(deep.promote_exact(256).is_err());
        assert_eq!(deep.promote_exact(512).unwrap(), deep);
        assert!((-0.0_f64).promote_exact(128).unwrap().is_sign_negative());
        let signed = DoubleFloat::from(-0.0);
        let promoted = signed.promote_exact(128).unwrap();
        assert!(promoted.is_sign_negative());
        assert!(signed.round_rescued(&promoted).negative_sign());
    }

    #[test]
    fn verified_batch_keeps_dual_axes_and_invalid_lanes_separate() {
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        let plan = ReductionPlan::new(settings).unwrap();
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = PrecisionRescueWorkspace::new_dual(
            &plan,
            0.0,
            RescueSettings {
                trigger: RescueTrigger::VerifyAllSuccessful,
                ..RescueSettings::default()
            },
            shape,
        )
        .unwrap();
        let c = |re| Complex::new(re, 0.0);
        let mut values = vec![c(0.0); 3 * 6 * 3];
        for lane in 0..3 {
            values[lane * 18 + 12] = c(2.0 + lane as f64);
            values[lane * 18 + 13] = c(1.0);
            values[lane * 18 + 15] = c(3.0);
        }
        values[2 * 18 + 15] = c(-1.0);
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 3,
                values,
            })
            .unwrap();
        assert_eq!(workspace.last_execution().requested_rows, 2);
        assert_eq!(workspace.last_execution().accepted_rows, 2);
        assert_eq!(workspace.last_execution().working.input_rows, 2);
        assert_eq!(workspace.last_execution().verification.input_rows, 2);
        for lane in 0..2 {
            let mass = 2.0 + lane as f64;
            let output = result.lane(lane).unwrap();
            assert_eq!(&output[..3], &[c(mass), c(1.0), c(0.0)]);
            assert!((output[10].re - (9.0 / mass).ln()).abs() < 1e-13);
            assert!((output[11].re + 0.5 / mass).abs() < 1e-13);
            assert!(matches!(
                workspace.last_rescue_statuses()[lane],
                RescueStatus::Converged { .. }
            ));
        }
        assert!(matches!(result.lane(2), Err(LaneStatus::InvalidInput(_))));
        assert_eq!(
            workspace.last_rescue_statuses()[2],
            RescueStatus::NotRequested
        );
        let empty = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 3,
                values: vec![],
            })
            .unwrap();
        assert_eq!(empty.rows(), 0);
        assert_eq!(workspace.last_execution().requested_rows, 0);
    }

    #[test]
    fn generalized_triangle_rescue_matches_every_reference_coefficient() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../../validation/reference/fixtures.json",))
                .unwrap();
        let case = fixtures["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["input"]["id"] == "C0_power2_1_1_rank2")
            .unwrap();
        let input = &case["input"];
        let propagators = input["propagators"].as_array().unwrap();
        let mut settings = ReductionSettings::new(3, 2);
        settings.algorithm = Algorithm::Ibp;
        settings.powers = propagators
            .iter()
            .map(|propagator| propagator["power"].as_u64().unwrap() as u16)
            .collect();
        let plan = ReductionPlan::new(settings).unwrap();
        let mut values = Vec::new();
        let mut scale = input["mu"].as_f64().unwrap();
        for propagator in propagators {
            for value in propagator["shift"].as_array().unwrap() {
                let value = value.as_f64().unwrap();
                scale = scale.max(value.abs());
                values.push(Complex::new(value, 0.0));
            }
        }
        for propagator in propagators {
            let mass = &propagator["mass_squared"];
            let value = Complex::new(mass[0].as_f64().unwrap(), mass[1].as_f64().unwrap());
            scale = scale.max(value.re.hypot(value.im).sqrt());
            values.push(value);
        }
        values.push(Complex::new(input["mu"].as_f64().unwrap(), 0.0));
        let mut workspace = PrecisionRescueWorkspace::new(
            &plan,
            0.0,
            RescueSettings {
                trigger: RescueTrigger::VerifyAllSuccessful,
                ..RescueSettings::default()
            },
        )
        .unwrap();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            })
            .unwrap();
        let actual = result.lane(0).unwrap();
        let expected = case["expected"]["moments"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len() * 4);
        for (moment, slots) in expected.iter().enumerate() {
            let natural = scale.powi(-4 + plan.basis()[moment].rank() as i32);
            for slot in 0..4 {
                let re = slots[slot][0].as_f64().unwrap();
                let im = slots[slot][1].as_f64().unwrap();
                let value = &actual[4 * moment + slot];
                assert!(
                    (value.re - re).hypot(value.im - im) <= 1e-10 * (natural + re.hypot(im)),
                    "moment {moment}, Laurent slot {slot}: {value:?} vs ({re},{im})"
                );
            }
        }
        assert_eq!(workspace.last_execution().accepted_rows, 1);
    }

    #[test]
    #[ignore = "full high-rank high-precision qualification; run explicitly"]
    fn five_point_rank_six_verifies_the_preserved_precision_discrepancy() {
        verify_highpoint_precision_discrepancy(6);
    }

    #[test]
    #[ignore = "full high-rank high-precision qualification; run explicitly"]
    fn five_point_rank_seven_verifies_every_component_at_two_precisions() {
        verify_highpoint_precision_discrepancy(7);
    }

    fn verify_highpoint_precision_discrepancy(rank: u16) {
        let started = std::time::Instant::now();
        let mut settings = ReductionSettings::new(5, rank);
        settings.algorithm = Algorithm::PassarinoVeltman;
        eprintln!("preparing the complete rank-{rank} pentagon plan");
        let plan = ReductionPlan::new(settings).unwrap();
        eprintln!(
            "plan prepared after {:?}: {} moments, {} aliases, {} master leaves; preparing three precision workspaces",
            started.elapsed(),
            plan.basis.len(),
            plan.aliases.len(),
            plan.leaves.len(),
        );
        let mut values: Vec<_> = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
        .into_iter()
        .flatten()
        .map(|value| Complex::new(value, 0.0))
        .collect();
        values.extend([4.0, 9.0, 16.0, 25.0, 36.0, 3.0].map(|value| Complex::new(value, 0.0)));
        let mut workspace = PrecisionRescueWorkspace::new(
            &plan,
            0.0,
            RescueSettings {
                trigger: RescueTrigger::VerifyAllSuccessful,
                ..RescueSettings::default()
            },
        )
        .unwrap();
        eprintln!(
            "workspaces prepared after {:?}; evaluating the batch",
            started.elapsed()
        );
        workspace.primary.set_profiling(true);
        workspace.working.set_profiling(true);
        workspace.verification.set_profiling(true);
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            })
            .unwrap();
        eprintln!("batch evaluated after {:?}", started.elapsed());
        eprintln!(
            "execution stages: primary={:?}, working={:?}, verification={:?}",
            workspace.primary.last_profile(),
            workspace.working.last_profile(),
            workspace.verification.last_profile(),
        );
        eprintln!(
            "execution diagnostics: primary={:?}, working={:?}, verification={:?}",
            workspace.primary.last_diagnostics(),
            workspace.working.last_diagnostics(),
            workspace.verification.last_diagnostics(),
        );
        eprintln!(
            "primary status: {:?}; precision status: {:?}; batch counters: {:?}",
            workspace.last_primary_statuses()[0],
            workspace.last_rescue_statuses()[0],
            workspace.last_execution(),
        );
        let index = plan
            .basis()
            .iter()
            .position(|moment| moment.0 == [5, 0, 0, 1])
            .unwrap();
        let value = result.lane(0).unwrap()[4 * index + 3];
        eprintln!("independent-oracle coefficient [5,0,0,1]: {value:?}");
        // Independently confirmed by Feynman-simplex quadrature (reported
        // error 2.77e-15), as well as the exact-coefficient/native-master probe
        // at 128/256 bits. Provenance: validation/pv-high-rank/
        // pentagon-simplex-oracle.json and precision-probe.json.
        let expected = -2.43837893544712293939249874984327534762e-3;
        assert!(value.re.abs() < 1e-13);
        assert!((value.im - expected).abs() < 1e-13, "{value:?}");
        assert_eq!(workspace.last_execution().accepted_rows, 1);
    }
}
