//! Exact selected Cartesian constraints for conservative legacy snapping and
//! checked supplied-G coordinate identities in every requested Taylor component.
//!
//! Symbolica selects the existing call-site branches and prepares every
//! polynomial/Dual product. The existing exact-zero certificate proves equality
//! on represented inputs; this module only packs lanes and selects predicates.
//! No shifted geometry, derivative projection or alternate metric is invented.

use super::*;
use crate::snapping::{SnapTarget, CartesianConstraintKind};

#[derive(Clone)]
pub(super) struct PreparedLegacyCompatibility {
    selectors: ExpressionEvaluator<Complex<Rational>>,
    certificate: ExactZeroConstraintPlan,
    labels: Vec<String>,
    kinds: Vec<CartesianConstraintKind>,
    parameters: usize,
    masses: std::ops::Range<usize>,
}

impl PreparedLegacyCompatibility {
    pub(super) fn prepare<D: DualNumberStructure>(
        plan: &ReductionPlan,
        stability: &StabilityPlan,
        dual: Option<&D>,
    ) -> Result<Option<Self>, ReductionError> {
        let queries = stability.snapping_compatibility();
        if queries.is_empty() {
            return Ok(None);
        }
        let mut parameters = plan.parameters.clone();
        // Real mass jets are packed exactly from the original input. CONJ is
        // never smuggled into a polynomial certificate as an opaque variable.
        let real_masses: Vec<_> = plan
            .layout
            .masses
            .clone()
            .map(|index| {
                Atom::var(symbolica::symbol!(format!(
                    "iregi_legacy_compatibility::mass_real_{index}"
                )))
            })
            .collect();
        parameters.extend(real_masses.iter().cloned());
        let mut selectors = Vec::new();
        let mut residuals = Vec::new();
        let mut labels = Vec::new();
        let mut kinds = Vec::new();
        for (query, active) in queries {
            for (choice, target) in &query.choices {
                selectors.push(symbolica::atom::Symbol::IF.call((active, choice, Atom::Zero)));
                residuals.push(match *target {
                    SnapTarget::Zero => query.raw.clone(),
                    SnapTarget::RealMass(index) => &query.raw - &real_masses[index],
                });
                labels.push(format!("{}, target {target:?}", query.label));
                kinds.push(query.kind);
            }
        }
        let certificate = match dual {
            Some(dual) => ExactZeroConstraintPlan::new_dual(&parameters, &residuals, dual),
            None => ExactZeroConstraintPlan::new(&parameters, &residuals),
        }?;
        // Predicates use only primal branch semantics. Their selected residuals
        // are subsequently checked with the complete unchanged Dual layout.
        let selectors = compile_exact::<D>(
            &selectors,
            &plan.parameters,
            stability.snapping_aliases(),
            None,
        )?;
        Ok(Some(Self {
            selectors,
            certificate,
            labels,
            kinds,
            parameters: plan.parameters.len(),
            masses: plan.layout.masses.clone(),
        }))
    }
}

#[derive(Clone)]
pub(super) struct LegacyCompatibilityWorkspace<T: NativeFloat> {
    selectors: BatchExpressionEvaluator<Complex<T>>,
    certificate: ExactZeroConstraintWorkspace,
    labels: Vec<String>,
    kinds: Vec<CartesianConstraintKind>,
    parameters: usize,
    masses: std::ops::Range<usize>,
    primal_inputs: Vec<Complex<T>>,
    choices: Vec<Complex<T>>,
    exact_inputs: Vec<Complex<Rational>>,
    selected: Vec<Vec<(usize, usize)>>,
    lanes: Vec<usize>,
}

impl<T: NativeFloat> LegacyCompatibilityWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn instantiate(prepared: &PreparedLegacyCompatibility, prototype: &T) -> Self {
        Self {
            selectors: map_exact(&prepared.selectors, prototype),
            certificate: prepared.certificate.instantiate(),
            labels: prepared.labels.clone(),
            kinds: prepared.kinds.clone(),
            parameters: prepared.parameters,
            masses: prepared.masses.clone(),
            primal_inputs: Vec::new(),
            choices: Vec::new(),
            exact_inputs: Vec::new(),
            selected: Vec::new(),
            lanes: Vec::new(),
        }
    }

    pub(super) fn failures(
        &mut self,
        input: &[Complex<T>],
        rows: usize,
        components: usize,
        prototype: &T,
    ) -> Result<Vec<(usize, LaneStatus)>, ReductionError> {
        self.primal_inputs.clear();
        self.exact_inputs.clear();
        self.selected.clear();
        self.lanes.clear();
        let input_width = checked_mul(self.parameters, components)?;
        if components == 0 || input.len() != checked_mul(rows, input_width)? {
            return Err(ReductionError::InvalidInput(
                "selected Cartesian compatibility input layout mismatch".into(),
            ));
        }
        self.primal_inputs
            .extend(input.chunks_exact(components).map(|jet| jet[0].clone()));
        self.choices.resize(
            checked_mul(rows, self.labels.len())?,
            Complex::new(prototype.zero(), prototype.zero()),
        );
        self.selectors
            .evaluate_batch(&self.primal_inputs, &mut self.choices, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
        let mut failures = Vec::new();
        for (lane, row) in input.chunks_exact(input_width).enumerate() {
            let mut selected = Vec::new();
            let mut malformed = false;
            for (constraint, value) in self.choices
                [lane * self.labels.len()..(lane + 1) * self.labels.len()]
                .iter()
                .enumerate()
            {
                if !finite(value)
                    || !value.im.is_zero()
                    || (!value.re.is_zero() && value.re != value.re.one())
                {
                    failures.push((lane, LaneStatus::Unstable(
                        "selected Cartesian compatibility branch cannot be classified at working precision".into())));
                    malformed = true;
                    break;
                }
                if !value.re.is_zero() {
                    selected.extend((0..components).map(|component| (constraint, component)));
                }
            }
            if malformed || selected.is_empty() {
                continue;
            }
            let start = self.exact_inputs.len();
            for value in row {
                let (Some(re), Some(im)) = (value.re.exact_rational(), value.im.exact_rational())
                else {
                    malformed = true;
                    break;
                };
                self.exact_inputs.push(Complex::new(re, im));
            }
            if !malformed {
                for argument in self.masses.clone() {
                    for value in &row[argument * components..(argument + 1) * components] {
                        let Some(re) = value.re.exact_rational() else {
                            malformed = true;
                            break;
                        };
                        self.exact_inputs.push(Complex::new(re, Rational::from(0)));
                    }
                    if malformed {
                        break;
                    }
                }
            }
            if malformed {
                self.exact_inputs.truncate(start);
                failures.push((
                    lane,
                    LaneStatus::Unstable(
                        "selected Cartesian compatibility cannot represent input jets exactly".into(),
                    ),
                ));
            } else {
                self.selected.push(selected);
                self.lanes.push(lane);
            }
        }
        if self.lanes.is_empty() {
            return Ok(failures);
        }
        let certificates = self.certificate.check_selected_batch(
            &self.exact_inputs,
            self.lanes.len(),
            &self.selected,
        )?;
        for (lane, certificate) in self.lanes.iter().copied().zip(certificates) {
            match certificate {
                ExactZeroCertificate::Zero => {}
                ExactZeroCertificate::Nonzero {
                    constraint,
                    component,
                } => {
                    let reason = match self.kinds[constraint] {
                        CartesianConstraintKind::LegacySnap => "legacy on-shell snap changes highpoint Cartesian geometry",
                        CartesianConstraintKind::CoordinateIdentity => "selected Cartesian coordinate identity is incompatible with the full Taylor geometry",
                    };
                    failures.push((lane, LaneStatus::InvalidInput(format!(
                        "{reason}: {}; Taylor component {component}; input preserved, no projection",
                        self.labels[constraint]))));
                }
                ExactZeroCertificate::Unresolved { working_bits } => {
                    failures.push((lane, LaneStatus::Unstable(format!(
                        "selected Cartesian geometry compatibility unresolved at {working_bits} certificate bits; input preserved"))));
                }
            }
        }
        Ok(failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ReductionSettings,
        snapping::{LegacySnapping, exact_threshold},
    };
    use symbolica::domains::{
        dual::HyperDual,
        float::{Float, SingleFloat},
    };

    fn c<T>(value: T, zero: T) -> Complex<T> {
        Complex::new(value, zero)
    }

    // A scalar A0 structural shell keeps these transport tests small. It is
    // deliberately NOT a claim of highpoint integration: only the original
    // input layout, demanded query graph, and exact pre-master guard execute.
    fn query_plan(unused_change: bool) -> ReductionPlan {
        let mut plan = ReductionPlan::new(ReductionSettings::new(1, 0)).unwrap();
        let masses = plan.parameters[plan.layout.masses.clone()].to_vec();
        let mut policy = LegacySnapping::new(&masses, "1/8").unwrap();
        let raw = plan.parameters[0].pow(2) - plan.parameters[1].pow(2);
        let used = policy.snap_scalar_product(raw.clone(), "physical relative norm");
        if unused_change {
            policy.snap_scalar_product(
                raw + Atom::num(Rational::from((1, 1024))),
                "discarded query",
            );
        }
        plan.moments[0] = [Atom::Zero, Atom::Zero, Atom::Zero, used];
        plan.aliases = policy.aliases().to_vec();
        plan.snapping_aliases = policy.aliases().to_vec();
        plan.snapping_diagnostics = policy.diagnostics().to_vec();
        plan.snapping_threshold = Some(exact_threshold("1/8").unwrap());
        plan.snapping_compatibility = policy.compatibility_queries().to_vec();
        plan
    }

    fn prepared<D: DualNumberStructure>(
        dual: Option<&D>,
        unused_change: bool,
    ) -> PreparedLegacyCompatibility {
        let plan = query_plan(unused_change);
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let result = PreparedLegacyCompatibility::prepare(&plan, &stability, dual)
            .unwrap()
            .unwrap();
        assert!(
            result
                .labels
                .iter()
                .all(|label| !label.contains("discarded"))
        );
        result
    }

    #[test]
    fn demanded_exact_mass_and_complex_width_preserve_geometry_but_changed_snap_fails() {
        let prepared = prepared::<HyperDual<u8>>(None, true);
        let mut workspace = LegacyCompatibilityWorkspace::instantiate(&prepared, &0.0);
        let mut first = vec![
            c(2.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(4.0, -0.5),
            c(3.0, 0.0),
        ];
        let original = first.clone();
        let mut changed = first.clone();
        changed[0].re += 1.0 / 1024.0;
        first.extend(changed);
        let failures = workspace.failures(&first, 2, 1, &0.0).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, 1);
        assert!(matches!(&failures[0].1,LaneStatus::InvalidInput(reason)
            if reason.contains("Cartesian geometry") && reason.contains("component 0")));
        assert_eq!(&first[..6], original.as_slice());
        let mut independent = workspace.clone();
        assert!(
            independent
                .failures(&original, 1, 1, &0.0)
                .unwrap()
                .is_empty()
        );
        assert!(workspace.failures(&[], 0, 1, &0.0).unwrap().is_empty());
        assert!(
            workspace
                .failures(&original, 1, 1, &0.0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn zero_snap_checks_derivative_components_and_keeps_neighbor_lane_independent() {
        let dual = HyperDual::<u8>::new(vec![vec![0], vec![1]]);
        let prepared = prepared(Some(&dual), false);
        let mut workspace = LegacyCompatibilityWorkspace::instantiate(&prepared, &0.0);
        // Both primal vectors are lightlike. Equal tangent coefficients keep
        // the first lane on the exact variety; the second is transverse.
        let tangent = vec![1., 1., 1., 1., 0., 0., 0., 0., 4., 0., 3., 0.]
            .into_iter()
            .map(|v| c(v, 0.))
            .collect::<Vec<_>>();
        let mut transverse = tangent.clone();
        transverse[3].re = 0.;
        let mut rows = tangent.clone();
        rows.extend(transverse);
        let failures = workspace.failures(&rows, 2, 2, &0.).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, 1);
        assert!(matches!(&failures[0].1,LaneStatus::InvalidInput(reason)
            if reason.contains("component 1")));
        assert!(workspace.failures(&tangent, 1, 2, &0.).unwrap().is_empty());
    }

    #[test]
    fn exact_selected_mass_checks_every_mixed_and_second_component() {
        let shape = vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![2, 0],
            vec![1, 1],
            vec![0, 2],
        ];
        let dual = HyperDual::<u8>::new(shape);
        let prepared = prepared(Some(&dual), false);
        let mut workspace = LegacyCompatibilityWorkspace::instantiate(&prepared, &0.);
        // Constant exact on-shell geometry. A nonprimal mass perturbation
        // changes the selected mass branch even though every primal agrees.
        let mut row = vec![c(0., 0.); 36];
        row[0] = c(2., 0.);
        row[24] = c(4., -0.5);
        row[30] = c(3., 0.);
        assert!(workspace.failures(&row, 1, 6, &0.).unwrap().is_empty());
        for component in 1..6 {
            let mut changed = row.clone();
            changed[24 + component].re = 1.;
            let failures = workspace.failures(&changed, 1, 6, &0.).unwrap();
            assert_eq!(failures.len(), 1);
            assert!(matches!(&failures[0].1,LaneStatus::InvalidInput(reason)
                if reason.contains(&format!("component {component}"))));
        }
        // Imaginary width derivatives do not alter Re(m²).
        row[28].im = 1.;
        assert!(workspace.failures(&row, 1, 6, &0.).unwrap().is_empty());
    }

    #[test]
    fn beyond_binary64_geometry_changes_are_not_rounded_into_compatibility() {
        let prepared = prepared::<HyperDual<u8>>(None, false);
        let zero = Float::new(128);
        let f = |v: i64| Float::with_val(128, v);
        let mut row = vec![
            c(f(2), zero.clone()),
            c(f(0), zero.clone()),
            c(f(0), zero.clone()),
            c(f(0), zero.clone()),
            c(f(4), zero.clone()),
            c(f(3), zero.clone()),
        ];
        let mut workspace = LegacyCompatibilityWorkspace::instantiate(&prepared, &zero);
        assert!(workspace.failures(&row, 1, 1, &zero).unwrap().is_empty());
        let tiny = Rational::from((1, 2)).pow(70);
        row[0].re += Float::from_rational(&zero, &tiny);
        let failures = workspace.failures(&row, 1, 1, &zero).unwrap();
        assert_eq!(failures.len(), 1);
        assert!(matches!(failures[0].1, LaneStatus::InvalidInput(_)));
    }
    #[test]
    fn coordinate_constraints_without_legacy_monitor_check_complete_ordered_mixed_jets() {
        use crate::snapping::SnapCompatibilityQuery;
        let mut plan = query_plan(false);
        let owner = Atom::var(symbolica::symbol!("iregi_coordinate_guard_test::selected"));
        plan.aliases.push((owner.clone(), Atom::num(1)));
        plan.moments[0][3] = symbolica::atom::Symbol::IF.call((&owner, Atom::num(1), Atom::Zero));
        plan.snapping_diagnostics.clear();
        plan.snapping_threshold = None;
        plan.snapping_zero_image = None;
        plan.snapping_zero_image_required = false;
        plan.snapping_compatibility = vec![SnapCompatibilityQuery {
            kind: CartesianConstraintKind::CoordinateIdentity,
            output: owner,
            raw: plan.parameters[0].pow(2) - plan.parameters[1].pow(2),
            label: "synthetic selected coordinate determinant".into(),
            choices: vec![(Atom::num(1), SnapTarget::Zero)],
        }];
        let stability = StabilityPlan::prepare(&plan).unwrap();
        assert!(stability.snapping_threshold().is_none());
        assert_eq!(stability.snapping_compatibility().len(), 1);
        let shape = HyperDual::<u8>::new(vec![vec![0,0],vec![0,1],vec![1,0],vec![1,1]]);
        let prepared = PreparedLegacyCompatibility::prepare(&plan, &stability, Some(&shape)).unwrap().unwrap();
        let mut workspace = LegacyCompatibilityWorkspace::instantiate(&prepared, &0.0);
        let mut tangent = vec![c(0.,0.); 24];
        for (component, value) in [1.,1.,2.,3.].into_iter().enumerate() {
            tangent[component] = c(value,0.);
            tangent[4+component] = c(value,0.);
        }
        tangent[16] = c(4.,0.); tangent[20] = c(3.,0.);
        let mut transverse = tangent.clone();transverse[7].re += 1.;
        let rows = [tangent.clone(),transverse,tangent.clone()].concat();
        let failures = workspace.failures(&rows,3,4,&0.).unwrap();
        assert_eq!(failures.len(),1);assert_eq!(failures[0].0,1);
        assert!(matches!(&failures[0].1,LaneStatus::InvalidInput(reason)
            if reason.contains("coordinate identity") && reason.contains("component 3")
                && !reason.contains("legacy on-shell")));
        assert!(workspace.failures(&tangent,1,4,&0.).unwrap().is_empty());
    }

    #[test]
    fn incompatible_snap_is_rejected_before_any_actual_scalar_master_request() {
        let mut plan = query_plan(false);
        // Keep an actual A0 master in the output graph. The accepted call must
        // perform real master work; rejection cannot be explained by no work.
        let master = plan.leaves[0].slots[3].clone();
        plan.moments[0][3] = &plan.moments[0][3] * master;
        let mut workspace = Workspace::new(&plan, 0.0).unwrap();
        let make = |time| KinematicsBatch {
            rows: 1,
            components: 1,
            values: [time, 0., 0., 0., 4., 3.]
                .into_iter()
                .map(|v| c(v, 0.))
                .collect(),
        };
        let good = make(2.);
        let result = workspace.reduce_batch(&good).unwrap();
        assert_eq!(result.statuses, [LaneStatus::Success]);
        assert!(workspace.last_execution().master_rows > 0);
        let bad = make(2. + 1. / 1024.);
        let before = bad.values.clone();
        let result = workspace.reduce_batch(&bad).unwrap();
        assert!(
            matches!(&result.statuses[0],LaneStatus::InvalidInput(reason)
            if reason.contains("Cartesian geometry"))
        );
        assert_eq!(bad.values, before);
        assert!(result.lane(0).is_err());
        assert_eq!(workspace.last_execution().master_rows, 0);
        assert_eq!(workspace.last_execution().master_family_batches, 0);
        assert_eq!(workspace.last_execution().fallback_rows, 0);
        assert_eq!(result.fallback_used, [false]);
        let result = workspace.reduce_batch(&good).unwrap();
        assert_eq!(result.statuses, [LaneStatus::Success]);
        assert!(workspace.last_execution().master_rows > 0);
    }
}
