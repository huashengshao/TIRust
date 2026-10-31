//! COLLIER-style UV and IR regulator parameters for returned tensor arrays.
//!
//! Every engine returns, per Cartesian moment, the Laurent slots
//! `[UV single, IR single, IR double, finite]` at the lane's scale `mu`, in
//! the selected [`Convention`]. COLLIER (Denner, Dittmaier, Hofer,
//! arXiv:1604.06792, sect. 2.2) instead returns one number per coefficient,
//!
//! `T = T_fin(mu_UV^2, mu_IR^2) + a_UV Delta_UV + a_IR2 (Delta_IR2 + Delta_IR1 ln mu_IR^2) + a_IR1 Delta_IR1`,
//!
//! with separate UV and IR scales and numerical values `Delta` standing in
//! for the poles (`SetMuUV2_cll`, `SetMuIR2_cll`, `SetDeltaUV_cll`,
//! `SetDeltaIR_cll`; all zero by default, giving the finite part). With
//! `L_U = ln(mu_UV^2/mu^2)` and `L_I = ln(mu_IR^2/mu^2)`, [`Regulators`] maps
//! the slots of each moment to
//!
//! - UV single: `uv`;
//! - IR single: `ir1 + ir2 L_I`;
//! - IR double: `ir2`;
//! - finite: `fin + uv L_U + ir1 L_I + ir2 L_I^2/2`
//!   `+ Delta_UV uv + Delta_IR1 (ir1 + ir2 L_I) + Delta_IR2 ir2`,
//!
//! so the finite slot holds COLLIER's number and the pole slots their
//! coefficients at the new scales (the poles are not double counted when the
//! Deltas are zero). [`RegulatorNormalization::Collier`] (with
//! [`Convention::OneLoop`] only) also removes `pi^2/6 ir2` from the finite
//! part: COLLIER factors `c(eps) = Gamma(1+eps) (4 pi)^eps` out of its poles,
//! OneLOop `(4 pi)^eps Gamma(1+eps) Gamma(1-eps)^2 / Gamma(1-2 eps)`
//! (COLLIER manual, sect. 2.2: `Delta_IR2 -> Delta_IR2 + pi^2/6`).
use iregi_masters::NativeFloat;
use symbolica::domains::float::{Complex, Float, Real};

use crate::{BatchResult, Convention, KinematicsBatch, LaneStatus, ReductionError};

/// Normalization of the regulated output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegulatorNormalization {
    /// The pole coefficients and finite part of the selected [`Convention`].
    #[default]
    Convention,
    /// COLLIER's `c(eps)` normalization (requires [`Convention::OneLoop`]).
    Collier,
}

/// COLLIER-style regulator parameters (see the module documentation). The
/// default (no scales, zero Deltas, convention normalization) leaves results
/// unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Regulators {
    /// UV scale `mu_UV^2`; `None` keeps the lane's `mu^2`.
    pub mu_uv_squared: Option<f64>,
    /// IR scale `mu_IR^2`; `None` keeps the lane's `mu^2`.
    pub mu_ir_squared: Option<f64>,
    pub delta_uv: f64,
    pub delta_ir1: f64,
    pub delta_ir2: f64,
    pub normalization: RegulatorNormalization,
}

/// The binary64 parameter `x` exactly at the working precision of `prototype`.
fn exact<T: NativeFloat>(prototype: &T, x: f64) -> T {
    match (&x as &dyn std::any::Any).downcast_ref::<T>() {
        Some(value) => value.clone(),
        None => Float::with_val(53, x)
            .try_to_rational()
            .map_or_else(|| prototype.zero(), |rational| prototype.from_rational(&rational)),
    }
}

impl Regulators {
    fn validate(&self, convention: Convention) -> Result<(), ReductionError> {
        for (name, value) in [("delta_uv", self.delta_uv), ("delta_ir1", self.delta_ir1), ("delta_ir2", self.delta_ir2)] {
            if !value.is_finite() {
                return Err(ReductionError::InvalidInput(format!("regulator {name} must be finite")));
            }
        }
        for (name, scale) in [("mu_uv_squared", self.mu_uv_squared), ("mu_ir_squared", self.mu_ir_squared)] {
            if scale.is_some_and(|s| !(s.is_finite() && s > 0.0)) {
                return Err(ReductionError::InvalidInput(format!("regulator {name} must be positive and finite")));
            }
        }
        if self.normalization == RegulatorNormalization::Collier && convention != Convention::OneLoop {
            return Err(ReductionError::Unsupported(
                "COLLIER regulator normalization is defined relative to Convention::OneLoop".into(),
            ));
        }
        Ok(())
    }

    /// Whether applying these parameters changes nothing.
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// Apply to every successful lane of `result` (the batch it was computed
    /// from supplies each lane's `mu`, its last argument). Ordinary numbers
    /// only: Dual components of `mu` would need derivatives of the logs.
    pub fn apply<T: NativeFloat>(
        &self,
        batch: &KinematicsBatch<T>,
        result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        if self.is_identity() {
            return Ok(());
        }
        self.validate(result.metadata().convention())?;
        if result.components != 1 || batch.components != 1 {
            return Err(ReductionError::Unsupported(
                "regulator parameters apply to ordinary (non-Dual) results".into(),
            ));
        }
        if batch.rows != result.rows() || batch.rows == 0 {
            return Ok(());
        }
        let width = batch.values.len() / batch.rows;
        let moments = result.basis.len();
        let prototype = batch.values[0].re.clone();
        let zero = prototype.zero();
        let half = prototype.one() / &prototype.from_usize(2);
        let pi = prototype.pi();
        let pi2_6 = pi.clone() * &pi / &prototype.from_usize(6);
        let deltas = [self.delta_uv, self.delta_ir1, self.delta_ir2].map(|x| exact(&prototype, x));
        for row in 0..batch.rows {
            // Best-effort arrays of refused lanes get the same convention.
            if result.statuses[row] != LaneStatus::Success && !result.is_best_effort(row) {
                continue;
            }
            let mu = batch.values[(row + 1) * width - 1].re.clone();
            let mu2 = mu.clone() * &mu;
            let log = |scale: Option<f64>| match scale {
                Some(s) => (exact(&prototype, s) / &mu2).log(),
                None => zero.clone(),
            };
            let (l_uv, l_ir) = (log(self.mu_uv_squared), log(self.mu_ir_squared));
            let lane = &mut result.values[row * moments * 4..(row + 1) * moments * 4];
            for slots in lane.chunks_exact_mut(4) {
                let scale = |z: &Complex<T>, k: &T| Complex::new(z.re.clone() * k, z.im.clone() * k);
                let add = |a: &Complex<T>, b: &Complex<T>| Complex::new(a.re.clone() + &b.re, a.im.clone() + &b.im);
                let (uv, ir1, ir2, fin) = (slots[0].clone(), slots[1].clone(), slots[2].clone(), slots[3].clone());
                let ir1_new = add(&ir1, &scale(&ir2, &l_ir));
                let mut finite = add(&fin, &scale(&uv, &l_uv));
                finite = add(&finite, &scale(&ir1, &l_ir));
                finite = add(&finite, &scale(&ir2, &(l_ir.clone() * &l_ir * &half)));
                if self.normalization == RegulatorNormalization::Collier {
                    finite = add(&finite, &scale(&ir2, &(zero.clone() - &pi2_6)));
                }
                finite = add(&finite, &scale(&uv, &deltas[0]));
                finite = add(&finite, &scale(&ir1_new, &deltas[1]));
                finite = add(&finite, &scale(&ir2, &deltas[2]));
                slots[1] = ir1_new;
                slots[3] = finite;
            }
        }
        Ok(())
    }
}
