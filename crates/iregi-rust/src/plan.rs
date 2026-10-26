//! Exact topology preparation. No sampled values are cached in a plan.

mod thin_seven;
mod affine_power;
pub(crate) use thin_seven::assemble_thin_seven_moments;

use crate::{
    basis::{Monomial, tensor_basis},
    convention::{convert_convention, convert_native_convention},
    ibp::{MasterLeaf, MasterLeafKind, ScalarCompiler, ScalarRequest},
    pv::PvCompiler,
    snapping::{LegacySnapping, SnapDiagnostic, exact_threshold},
    types::{
        Algorithm, InputLayout, InputPolicy, InvariantInput, ReductionError, ReductionSettings,
    },
};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, Mutex, OnceLock},
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, integer::Integer},
};

#[derive(Clone, Debug)]
pub struct ReductionPlan {
    pub(crate) execution_cache: crate::execution::ExactExecutionCache,
    variant_cache: StructuralPlanCache,
    pub(crate) settings: ReductionSettings,
    pub(crate) basis: Vec<Monomial>,
    pub(crate) parameters: Vec<Atom>,
    /// Private imported guarded operands: value, primal amplitude, mass dimension.
    pub(crate) external_primal_seeds: Vec<(Atom, Atom, i32)>,
    pub(crate) layout: InputLayout,
    pub(crate) distances: Vec<Vec<Atom>>,
    /// Polynomial distances in original inputs, before the optional on-shell
    /// policy. These certify numerical zeros without undoing intentional snaps.
    pub(crate) raw_distances: Vec<Vec<Atom>>,
    pub(crate) moments: Vec<[Atom; 4]>,
    pub(crate) leaves: Vec<MasterLeaf>,
    pub(crate) master_arguments: Vec<Vec<Atom>>,
    pub(crate) aliases: Vec<(Atom, Atom)>,
    /// Invariant-only definitions required by argument and diagnostic lowering.
    pub(crate) snapping_aliases: Vec<(Atom, Atom)>,
    pub(crate) snapping_diagnostics: Vec<SnapDiagnostic>,
    /// The effective threshold, absent for preserved and supplied-G inputs.
    pub(crate) snapping_threshold: Option<Atom>,
    /// A zero-threshold negative reference can violate the zero-distance
    /// diagonal assumed by the recurrence. Reject only demanded matrix paths.
    pub(crate) snapping_zero_image: Option<Atom>,
    pub(crate) snapping_zero_image_required: bool,
    /// Demanded highpoint snaps need an exact full-Taylor no-change proof.
    pub(crate) snapping_compatibility: Vec<crate::snapping::SnapCompatibilityQuery>,
    pub(crate) fallback: Option<Box<ReductionPlan>>,
    algorithm_used: Algorithm,
    preparation_notes: Vec<String>,
    pub(crate) special_kinematics: bool,
    pub(crate) analytic_jet_order: u16,
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<ZeroInvariantProfile>,
    /// A structural chart whose full-Taylor invariant constraints are checked
    /// by the private certified scalar entry point before numerical execution.
    certified_invariant_chart: Option<CertifiedInvariantChart>,
    /// Fixed constant-one pentagon chart, executable only after full input certification.
    certified_planar_chart: Option<usize>,
    certified_planar_mask: u8,
    certified_affine_chart: Option<CertifiedAffineChart>,
    /// Structural imported-child context; never sampled certificate state.
    thin_six_child_rank: Option<u16>,
    thin_seven_child_rank: Option<u16>,
}

/// Private proof context. A structural plan is executable only through its
/// checked wrapper, which recertifies every lane and every requested jet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CertifiedInvariantChart {
    Scalar([usize; 4]),
    CartesianTensor([usize; 4]),
}

impl CertifiedInvariantChart {
    fn basis(self) -> [usize; 4] {
        match self { Self::Scalar(basis) | Self::CartesianTensor(basis) => basis }
    }
    fn tensor_completion_allowed(self) -> bool {
        matches!(self, Self::CartesianTensor(_))
    }
}

/// Private structural proof context; fresh full-Taylor certificates live only
/// in the owning affine workspace, never in this plan/cache key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CertifiedAffineChart {
    pub key: crate::pv::AffineChartKey,
    pub inactive: Vec<usize>,
}

/// Exact constraints are structural: numerical values are never stored here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ZeroInvariantProfile {
    pub masses: Vec<bool>,
    pub distances: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PlanVariant {
    analytic_jet_order: u16,
    bubble_equal_masses: Option<bool>,
    zero_invariants: Option<ZeroInvariantProfile>,
    certified_planar_mask: u8,
    certified_affine_chart: Option<CertifiedAffineChart>,
    /// Structural imported-child context; never sampled certificate state.
    thin_six_child_rank: Option<u16>,
    thin_seven_child_rank: Option<u16>,
    certified_invariant_chart: Option<CertifiedInvariantChart>,
}

type StructuralPlanCache =
    Arc<Mutex<HashMap<PlanVariant, Arc<OnceLock<Result<Arc<ReductionPlan>, ReductionError>>>>>>;

impl ReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0, None, None, 0)
    }

    pub(crate) fn new_certified_affine(settings: ReductionSettings,
        context: CertifiedAffineChart) -> Result<Self,ReductionError> {
        Self::build_with_affine(settings,None,None,0,None,None,0,Some(context))
    }

    /// Private, bounded thin N6 plan. Imported child moments are already in
    /// native IREGI normalization and are never converted a second time.
    pub(crate) fn new_thin_six(settings: ReductionSettings, analytic_jet_order: u16) -> Result<Self, ReductionError> {
        use crate::types::{Convention, TensorWeights};
        let supplied = settings.invariants == InvariantInput::SuppliedScalarProducts;
        if settings.propagators != 6 || settings.rank > if supplied { 6 } else { 7 }
            || settings.powers != [1;6] || settings.dimension_shift != 0
            || settings.input_policy != InputPolicy::Preserve
            || (!supplied && (settings.convention != Convention::IregiNative
                || settings.tensor_weights != TensorWeights::Components))
            || settings.algorithm != Algorithm::PassarinoVeltman {
            return Err(ReductionError::Unsupported("private thin N6 requires unit powers, h=0, preserved inputs and PV; supplied products additionally require the checked tensor entry and rank<=6".into()));
        }
        let shifts: Vec<[Atom;4]> = (0..6).map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}")))).collect();
        let masses: Vec<_> = (0..6).map(|i| variable(&format!("m2_{i}"))).collect();
        let mut parameters: Vec<_> = shifts.iter().flatten().cloned().collect();
        let (scalar_products, distances): (Option<std::ops::Range<usize>>, Vec<Vec<Atom>>) = if supplied {
            let start=parameters.len();
            let mut products=vec![vec![Atom::new();6];6];
            for i in 0..6 { for j in i..6 {
                let value=variable(&format!("g{i}_{j}"));
                parameters.push(value.clone()); products[i][j]=value.clone(); products[j][i]=value;
            }}
            let distances=(0..6).map(|i|(0..6).map(|j|
                &products[i][i]+&products[j][j]-Atom::num(2)*&products[i][j]).collect()).collect();
            (Some(start..parameters.len()),distances)
        } else {
            (None,shifts.iter().map(|p| shifts.iter().map(|q| squared_difference(p,q)).collect()).collect())
        };
        let mass_start=parameters.len();
        parameters.extend(masses.iter().cloned()); parameters.push(variable("mu"));
        let layout = InputLayout { shifts:0..24, scalar_products, masses:mass_start..mass_start+6, scale:mass_start+6 };
        let mut compiler = PvCompiler::with_distances(shifts, masses, distances.clone(), true).map_err(ReductionError::Preparation)?;
        let weights = compiler.thin_six_weights().map_err(ReductionError::Preparation)?;
        let child_rank = settings.rank.saturating_sub(1);
        let child_basis = tensor_basis(child_rank);
        let child_values = 6 * 4 * child_basis.len();
        let values: Vec<_> = (0..child_values).map(|index| variable(&format!("thin_child_{index}"))).collect();
        let amplitudes: Vec<_> = (0..child_values).map(|index| variable(&format!("thin_amplitude_{index}"))).collect();
        let external_primal_seeds = thin_six_external_primal_seeds(&child_basis, &values, &amplitudes);
        parameters.extend(values.iter().cloned()); parameters.extend(amplitudes);
        let basis = tensor_basis(settings.rank);
        let mut moments = assemble_thin_six_moments(&basis, &child_basis, &weights, &values);
        // Children are already native raw moments: apply only the existing
        // final-output UV rule, never a second convention conversion.
        apply_final_uv_power_counting(&settings, &basis, &mut moments)?;
        if supplied {
            // Imported children remain native raw components. Convert exactly
            // once at this final parent, using the native-to-requested Symbolica adapter.
            for (moment,monomial) in moments.iter_mut().zip(&basis) {
                *moment=convert_native_convention(moment,*monomial,&settings);
            }
        }
        Ok(Self {
            execution_cache: Default::default(), variant_cache: Default::default(),
            settings, basis, parameters, external_primal_seeds, layout,
            raw_distances: distances.clone(), distances, moments,
            leaves: Vec::new(), master_arguments: Vec::new(), aliases: compiler.aliases().to_vec(),
            snapping_aliases:Vec::new(), snapping_diagnostics:Vec::new(), snapping_threshold:None,
            snapping_zero_image:None, snapping_zero_image_required:false, snapping_compatibility:Vec::new(), fallback:None,
            algorithm_used:Algorithm::PassarinoVeltman,
            preparation_notes:vec![if supplied {
                format!("Private packed checked-G N6 PV: original supplied products in parent weights and every packed N5 scalar argument; fresh complete G/p child proofs and planar masks precede child graph preparation; rank-{child_rank} native children converted only at final output")
            } else {format!("Private thin N6: one packed N5 rank-{child_rank} batch, retained primal operand envelopes; no quadrature route support")}],
            special_kinematics:true, analytic_jet_order, bubble_limit:None, zero_invariants:None,
            certified_invariant_chart:None, certified_planar_chart:None, certified_planar_mask:0, certified_affine_chart: None, thin_six_child_rank:Some(child_rank), thin_seven_child_rank:None,
        })
    }

    /// Only the certified scalar wrapper may prepare this chart. A plan holds
    /// no numerical certificate; that wrapper rechecks every input batch.
    pub(crate) fn new_certified_scalar(
        settings: ReductionSettings,
        basis: [usize; 4],
    ) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0, Some(CertifiedInvariantChart::Scalar(basis)), None, 0)
    }

    /// Private checked Cartesian chart: supplied G remains authoritative for
    /// scalar arguments, while the wrapper proves its full-jet compatibility
    /// with the unchanged Cartesian momenta before every numerical call.
    pub(crate) fn new_certified_tensor(
        settings: ReductionSettings,
        basis: [usize; 4],
    ) -> Result<Self, ReductionError> {
        if crate::certified_tensor::packed_settings_supported(&settings) {
            // Only the enclosing checked wrapper can reach this private plan;
            // its full G/p and Gram proofs still precede every numerical call.
            return Self::new_thin_six(settings,0);
        }
        Self::build(settings, None, None, 0,
            Some(CertifiedInvariantChart::CartesianTensor(basis)), None, 0)
    }

    /// Private fixed chart for the checked planar wrapper. Inputs must be
    /// recertified for every batch; this plan never contains sampled evidence.
    pub(crate) fn new_certified_planar(
        settings: ReductionSettings,
        omitted_row: usize,
    ) -> Result<Self, ReductionError> {
        Self::new_certified_planar_masked(settings, omitted_row, 0)
    }

    /// Only the checked wrapper may pass a mask; every marked coefficient's
    /// full requested cofactor jet must have a fresh exact-zero certificate.
    pub(crate) fn new_certified_planar_masked(
        settings: ReductionSettings, omitted_row: usize, mask: u8,
    ) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0, None, Some(omitted_row), mask)
    }

    /// An analytic Taylor neighborhood selected only for exact lightlike bubbles.
    /// `jet_order` bounds the total derivative degree of the caller's Dual layout.
    pub(crate) fn lightlike_bubble(
        settings: ReductionSettings,
        jet_order: u16,
        equal_masses: bool,
    ) -> Result<Self, ReductionError> {
        if settings.propagators != 2 {
            return Err(ReductionError::InvalidInput(
                "lightlike bubble requires two propagators".into(),
            ));
        }
        Self::build(
            settings,
            Some((jet_order, equal_masses)),
            None,
            jet_order,
            None,
            None,
            0,
        )
    }

    pub(crate) fn with_analytic_jet_order(&self, order: u16) -> Result<Arc<Self>, ReductionError> {
        self.variant(PlanVariant {
            analytic_jet_order: order,
            bubble_equal_masses: self.bubble_limit.map(|(_, equal)| equal),
            zero_invariants: self.zero_invariants.clone(),
            certified_planar_mask: self.certified_planar_mask, certified_affine_chart: self.certified_affine_chart.clone(),
            certified_invariant_chart: self.certified_invariant_chart,
            thin_six_child_rank: self.thin_six_child_rank,
            thin_seven_child_rank: self.thin_seven_child_rank,
        })
    }

    /// Prepare an exact invariant profile. The runtime must verify each selected
    /// constraint is zero in every Taylor component before dispatching a lane.
    pub(crate) fn with_zero_invariants(
        &self,
        mut profile: ZeroInvariantProfile,
    ) -> Result<Arc<Self>, ReductionError> {
        let n = self.settings.propagators;
        if profile.masses.len() != n
            || profile
                .distances
                .iter()
                .any(|&(i, j)| i >= n || j >= n || i == j)
        {
            return Err(ReductionError::InvalidInput(
                "invalid zero-invariant profile".into(),
            ));
        }
        for pair in &mut profile.distances {
            if pair.0 > pair.1 {
                *pair = (pair.1, pair.0);
            }
        }
        profile.distances.sort_unstable();
        profile.distances.dedup();
        self.variant(PlanVariant {
            analytic_jet_order: self.analytic_jet_order,
            bubble_equal_masses: None,
            zero_invariants: Some(profile),
            certified_planar_mask: self.certified_planar_mask, certified_affine_chart: self.certified_affine_chart.clone(),
            certified_invariant_chart: self.certified_invariant_chart,
            thin_six_child_rank: self.thin_six_child_rank,
            thin_seven_child_rank: self.thin_seven_child_rank,
        })
    }

    fn variant(&self, variant: PlanVariant) -> Result<Arc<Self>, ReductionError> {
        if variant.analytic_jet_order == self.analytic_jet_order
            && variant.bubble_equal_masses == self.bubble_limit.map(|(_, equal)| equal)
            && variant.zero_invariants == self.zero_invariants
            && variant.certified_affine_chart == self.certified_affine_chart
            && variant.certified_planar_mask == self.certified_planar_mask
            && variant.certified_invariant_chart == self.certified_invariant_chart
            && variant.thin_six_child_rank == self.thin_six_child_rank
            && variant.thin_seven_child_rank == self.thin_seven_child_rank
        {
            // Do not insert self into its own cache: even an immutable clone
            // would retain this cache and create an ownership cycle.
            return Ok(Arc::new(self.clone()));
        }
        // Imported affine-power graphs have their exact analytic order and
        // proof-selected source layout fixed by their owning wrapper. Never
        // rebuild one as an unrelated generic scalar/tensor plan.
        if !self.external_primal_seeds.is_empty()
            && self.thin_six_child_rank.is_none() && self.thin_seven_child_rank.is_none()
        {
            return Err(ReductionError::Unsupported(
                "affine-power imported plans require their owner to rebuild a structural variant".into()));
        }
        if variant.certified_affine_chart != self.certified_affine_chart {
            return Err(ReductionError::Unsupported("structural variant cannot change affine proof context".into()));
        }
        if variant.certified_invariant_chart != self.certified_invariant_chart {
            return Err(ReductionError::Unsupported(
                "structural variants cannot change the checked supplied-G proof context".into()));
        }
        if variant.thin_six_child_rank != self.thin_six_child_rank
            || variant.thin_seven_child_rank != self.thin_seven_child_rank
            || ((self.thin_six_child_rank.is_some() || self.thin_seven_child_rank.is_some())
                && (variant.bubble_equal_masses.is_some() || variant.zero_invariants.is_some()
                    || variant.certified_planar_mask != 0)) {
            return Err(ReductionError::Unsupported("thin high-point variants preserve imported-child topology and cannot accept invariant profiles".into()));
        }
        let entry = self
            .variant_cache
            .lock()
            .map_err(|_| ReductionError::Preparation("structural plan cache lock poisoned".into()))?
            .entry(variant.clone())
            .or_default()
            .clone();
        // No cache lock is held while preparing fallback/profile recurrences.
        // Every newly built child receives fresh caches, with no parent link.
        entry
            .get_or_init(|| {
                if self.thin_seven_child_rank.is_some() {
                    return Self::new_thin_seven(self.settings.clone(), variant.analytic_jet_order).map(Arc::new);
                }
                if self.thin_six_child_rank.is_some() {
                    // Imported-child parents contain no master/analytic
                    // recurrence. Their exact formulas and imported operands
                    // are independent of the requested Taylor expansion order.
                    // Preserve that immutable graph while each ordered shape
                    // gets its own execution and structural cache contexts.
                    let mut rebuilt = self.clone();
                    rebuilt.execution_cache = Default::default();
                    rebuilt.variant_cache = Default::default();
                    rebuilt.analytic_jet_order = variant.analytic_jet_order;
                    return Ok(Arc::new(rebuilt));
                }
                Self::build_with_affine(
                    self.settings.clone(),
                    variant
                        .bubble_equal_masses
                        .map(|equal| (variant.analytic_jet_order, equal)),
                    variant.zero_invariants,
                    variant.analytic_jet_order,
                    self.certified_invariant_chart,
                    self.certified_planar_chart,
                    variant.certified_planar_mask,
                    variant.certified_affine_chart.clone(),
                )
                .map(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| match error {
                ReductionError::InvalidInput(reason) => {
                    ReductionError::InvalidInput(reason.clone())
                }
                ReductionError::Preparation(reason) => ReductionError::Preparation(reason.clone()),
                ReductionError::Unsupported(reason) => ReductionError::Unsupported(reason.clone()),
                ReductionError::Evaluation(reason) => ReductionError::Evaluation(reason.clone()),
            })
    }

    fn build(
        settings: ReductionSettings, bubble_limit: Option<(u16,bool)>,
        zero_invariants: Option<ZeroInvariantProfile>, analytic_jet_order: u16,
        certified_invariant_chart: Option<CertifiedInvariantChart>,
        certified_planar_chart: Option<usize>, certified_planar_mask: u8,
    ) -> Result<Self, ReductionError> {
        Self::build_with_affine(settings,bubble_limit,zero_invariants,analytic_jet_order,
            certified_invariant_chart,certified_planar_chart,certified_planar_mask,None)
    }

    fn build_with_affine(
        settings: ReductionSettings,
        bubble_limit: Option<(u16, bool)>,
        zero_invariants: Option<ZeroInvariantProfile>,
        analytic_jet_order: u16,
        certified_invariant_chart: Option<CertifiedInvariantChart>,
        certified_planar_chart: Option<usize>,
    certified_planar_mask: u8,
        certified_affine_chart: Option<CertifiedAffineChart>,
    ) -> Result<Self, ReductionError> {
        if certified_planar_mask >= 31 || (certified_planar_mask != 0 && certified_planar_chart.is_none()) {
            return Err(ReductionError::InvalidInput("certified planar mask must contain only five child bits, retain a child, and belong to a checked chart".into()));
        }
        if let Some(chart) = &certified_affine_chart {
            if !crate::certified_affine::settings_supported(&settings)
                || chart.key.coordinate_rows.len()!=1
                || chart.key.coordinate_rows[0]>=4
                || chart.key.relative_columns!=[0,1]
                || chart.inactive.len()>=3
                || chart.inactive.iter().any(|&i|i>=3)
                || chart.inactive.windows(2).any(|p|p[0]>=p[1])
                || certified_invariant_chart.is_some() || certified_planar_chart.is_some()
                || certified_planar_mask!=0 || bubble_limit.is_some() || zero_invariants.is_some() {
                return Err(ReductionError::Unsupported("invalid private certified N3 affine context".into()));
            }
        }
        let n = settings.propagators;
        if n == 0 || settings.powers.len() != n || settings.powers.iter().any(|&p| p == 0) {
            return Err(ReductionError::InvalidInput(
                "provide a nonempty topology and one positive power per propagator".into(),
            ));
        }
        if settings.rank > 32 || n > 16 {
            return Err(ReductionError::Unsupported(
                "preparation budget exceeds 32 tensor indices or 16 propagators".into(),
            ));
        }
        if let Some(omitted) = certified_planar_chart {
            if !(crate::certified_planar::settings_supported(&settings)
                || crate::certified_planar::supplied_six_point_child_settings(&settings))
                || omitted >= 4
                || (certified_planar_mask != 0
                    && crate::certified_planar::generalized_scalar_settings(&settings))
                || zero_invariants.is_some()
                || bubble_limit.is_some()
                || certified_invariant_chart.is_some()
            {
                return Err(ReductionError::InvalidInput("certified planar charts require a supported preserved Cartesian N5 domain, an omitted coordinate row in 0..4, and no generalized cofactor mask".into()));
            }
        }
        // High-point pinching identities require a four-dimensional external
        // Gram matrix. Arbitrary supplied invariants, including Dual directions
        // transverse to that variety, do not satisfy those identities. Until
        // every Taylor component is checked, do not accept this input domain.
        if let Some(context) = certified_invariant_chart {
            let chart = context.basis();
            let valid_mode = match context {
                CertifiedInvariantChart::Scalar(_) => settings.rank == 0
                    && settings.algorithm == Algorithm::Ibp,
                CertifiedInvariantChart::CartesianTensor(_) =>
                    crate::certified_tensor::settings_supported(&settings),
            };
            if !(6..=7).contains(&n)
                || !valid_mode
                || settings.invariants != InvariantInput::SuppliedScalarProducts
                || chart.iter().any(|&index| index == 0 || index >= n)
                || chart.iter().enumerate().any(|(i, index)| chart[..i].contains(index))
            {
                return Err(ReductionError::InvalidInput(
                    "certified supplied-G charts require their checked scalar/tensor settings and four distinct nonzero propagator indices".into(),
                ));
            }
        }
        if settings.invariants == InvariantInput::SuppliedScalarProducts
            && n >= 6
            && certified_invariant_chart.is_none()
        {
            return Err(ReductionError::Unsupported(
                "supplied scalar products with six or more propagators require validated four-dimensional Gram constraints in every Taylor component; coordinate inputs remain supported".into(),
            ));
        }
        let requested_threshold = match &settings.input_policy {
            InputPolicy::Preserve => None,
            InputPolicy::LegacyOnShell { threshold } => Some(exact_threshold(threshold)?),
        };
        let snapping_threshold = if settings.invariants == InvariantInput::FromShifts {
            requested_threshold
        } else {
            None
        };
        if snapping_threshold.is_some() && n >= 5
            && !crate::snapping::highpoint_legacy_domain(&settings) {
            return Err(ReductionError::Unsupported(
                "highpoint legacy on-shell coordinates currently require N=5 or 6 and rank<=6; geometry-changing snaps are not admitted".into(),
            ));
        }
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..n).map(|i| variable(&format!("m2_{i}"))).collect();
        let snapping = match (&settings.input_policy, settings.invariants) {
            (InputPolicy::LegacyOnShell { threshold }, InvariantInput::FromShifts) => Some(
                Rc::new(RefCell::new(LegacySnapping::new(&masses, threshold)?)),
            ),
            _ => None,
        };
        let mu = variable("mu");
        let mut parameters: Vec<Atom> = shifts.iter().flatten().cloned().collect();
        let mut distances = vec![vec![Atom::new(); n]; n];
        let mut raw_distances = distances.clone();
        let scalar_products = match settings.invariants {
            InvariantInput::FromShifts => {
                for i in 0..n {
                    for j in i + 1..n {
                        let raw = squared_difference(&shifts[i], &shifts[j]);
                        raw_distances[i][j] = raw.clone();
                        raw_distances[j][i] = raw.clone();
                        distances[i][j] = if let Some(snapping) = &snapping {
                            snapping
                                .borrow_mut()
                                .snap_scalar_product(raw, format!("(p{i}-p{j})²"))
                        } else {
                            raw
                        };
                        distances[j][i] = distances[i][j].clone();
                    }
                }
                None
            }
            InvariantInput::SuppliedScalarProducts => {
                let start = parameters.len();
                let mut products = vec![vec![Atom::new(); n]; n];
                for i in 0..n {
                    for j in i..n {
                        let value = variable(&format!("g{i}_{j}"));
                        products[i][j] = value.clone();
                        products[j][i] = value.clone();
                        parameters.push(value);
                    }
                }
                for i in 0..n {
                    for j in i + 1..n {
                        distances[i][j] =
                            &products[i][i] + &products[j][j] - Atom::num(2) * &products[i][j];
                        distances[j][i] = distances[i][j].clone();
                        raw_distances[i][j] = distances[i][j].clone();
                        raw_distances[j][i] = distances[i][j].clone();
                    }
                }
                Some(start..parameters.len())
            }
        };
        let mass_start = parameters.len();
        parameters.extend(masses.iter().cloned());
        let scale = parameters.len();
        parameters.push(mu.clone());
        let layout = InputLayout {
            shifts: 0..4 * n,
            scalar_products,
            masses: mass_start..scale,
            scale,
        };
        let basis = tensor_basis(settings.rank);
        let ordinary = settings.dimension_shift == 0 && settings.powers.iter().all(|&p| p == 1);
        let special_kinematics = bubble_limit.is_some() || zero_invariants.is_some();
        let algorithm_used = if !ordinary || special_kinematics {
            Algorithm::Ibp
        } else {
            settings.algorithm
        };
        let mut preparation_notes = Vec::new();
        if matches!(settings.input_policy, InputPolicy::LegacyOnShell { .. }) {
            preparation_notes.push(if snapping.is_some() {
                "Legacy on-shell policy applies to individual scalar products; input momenta and masses are preserved".into()
            } else {
                "Supplied scalar products bypass the legacy on-shell policy, matching the original supplied-invariant interface".into()
            });
        }
        let checked_cartesian = matches!(certified_invariant_chart,
            Some(CertifiedInvariantChart::CartesianTensor(_)));
        let mut coordinate_constraints = Vec::new();
        let mut generalized_ibp_used = false;
        let (mut moments, leaves, mut aliases, algorithm_used) = if let Some(context) = &certified_affine_chart {
            let mut compiler=PvCompiler::with_distances(shifts.clone(),masses.clone(),distances.clone(),true)
                .map_err(ReductionError::Preparation)?;
            compiler.set_analytic_jet_order(analytic_jet_order).map_err(ReductionError::Preparation)?;
            let exponents:Vec<_>=basis.iter().map(|m|m.0).collect();
            let moments=compiler.compile_affine_descent_chart_moments(&context.key,&context.inactive,
                &exponents,crate::pv::AffineDescentBudget::default()).map_err(ReductionError::Preparation)?;
            preparation_notes.push(format!("Certified constant-one affine descent: original q, N3->same-rank bubbles, coordinate chart {:?}, inactive children {:?}; every selected row requires fresh full-Taylor rank/chart/cofactor certificates",context.key,context.inactive));
            (moments,compiler.leaves().to_vec(),compiler.aliases().to_vec(),Algorithm::PassarinoVeltman)
        } else if !ordinary
            && !special_kinematics
            && settings.algorithm != Algorithm::Ibp
            && !matches!(certified_invariant_chart, Some(CertifiedInvariantChart::CartesianTensor(_)))
        {
            let generalized = compile_generalized_pv(
                &settings,
                &basis,
                &shifts,
                &masses,
                &distances,
                analytic_jet_order,
                snapping.clone(),
                certified_planar_chart,
            );
            match generalized {
                Ok((moments, leaves, aliases, usage)) => {
                    generalized_ibp_used = usage.ibp != 0;
                    preparation_notes.push(format!(
                        "Generalized scalar-term dispatch: {} canonical PV requests, {} IBP requests; PV eligibility requires first power one and h >= sum(powers)-N",
                        usage.pv, usage.ibp,
                    ));
                    if n > 4 && certified_planar_chart.is_none() {
                        preparation_notes.push("Generalized canonical PV coefficients beyond four propagators are not qualified; retaining genuine IBP and shifted-master reduction, including D6 pentagons".into());
                    }
                    let used = if usage.pv == 0 {
                        Algorithm::Ibp
                    } else {
                        // A generalized PV primary may contain IBP terms which
                        // are ineligible for IBP2PAVE, as in original IREGI.
                        Algorithm::PassarinoVeltman
                    };
                    (moments, leaves, aliases, used)
                }
                Err(error) if settings.algorithm == Algorithm::Hybrid => {
                    preparation_notes.push(format!(
                        "Generalized PV preparation failed; using IBP: {error}"
                    ));
                    let (moments, leaves, aliases) = compile_ibp(
                        &settings, &basis, &shifts, &masses, &distances, None, None, None,
                        certified_planar_chart,
                    )?;
                    (moments, leaves, aliases, Algorithm::Ibp)
                }
                Err(error) => return Err(error),
            }
        } else {
            match algorithm_used {
                Algorithm::Ibp => {
                    match compile_ibp(
                        &settings,
                        &basis,
                        &shifts,
                        &masses,
                        &distances,
                        bubble_limit,
                        zero_invariants.as_ref(),
                        certified_invariant_chart.map(CertifiedInvariantChart::basis),
                        certified_planar_chart,
                    ) {
                        Ok((moments, leaves, aliases)) => {
                            (moments, leaves, aliases, Algorithm::Ibp)
                        }
                        Err(error)
                            if ordinary
                                && n >= 5
                                && !special_kinematics
                                && certified_invariant_chart.is_none_or(CertifiedInvariantChart::tensor_completion_allowed) =>
                        {
                            // Original IBP also completes its four-dimensional
                            // high-point tensor system with PV at zero dimension
                            // coefficients. This does not replace an independent
                            // higher-dimensional scalar integral by box masters.
                            preparation_notes
                                .push(format!("IBP tensor completion uses PV: {error}"));
                            let (moments, leaves, aliases) = compile_pv(
                                &basis,
                                &shifts,
                                &masses,
                                &distances,
                                analytic_jet_order,
                                snapping.clone(),
                                certified_planar_chart,
                                certified_planar_mask,
                                checked_cartesian.then_some(&mut coordinate_constraints),
                            )?;
                            (moments, leaves, aliases, Algorithm::PassarinoVeltman)
                        }
                        Err(error) => return Err(error),
                    }
                }
                Algorithm::PassarinoVeltman | Algorithm::Hybrid => {
                    let pv = compile_pv(
                        &basis,
                        &shifts,
                        &masses,
                        &distances,
                        analytic_jet_order,
                        snapping.clone(),
                        certified_planar_chart,
                        certified_planar_mask,
                        checked_cartesian.then_some(&mut coordinate_constraints),
                    );
                    match pv {
                        Ok((moments, leaves, aliases)) => {
                            (moments, leaves, aliases, Algorithm::PassarinoVeltman)
                        }
                        Err(error) if algorithm_used == Algorithm::Hybrid => {
                            preparation_notes
                                .push(format!("PV preparation failed; using IBP: {error}"));
                            let (moments, leaves, aliases) = compile_ibp(
                                &settings,
                                &basis,
                                &shifts,
                                &masses,
                                &distances,
                                bubble_limit,
                                zero_invariants.as_ref(),
                                certified_invariant_chart.map(CertifiedInvariantChart::basis),
                                certified_planar_chart,
                            )?;
                            (moments, leaves, aliases, Algorithm::Ibp)
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
        };
        let (snapping_aliases, snapping_diagnostics, snapping_zero_image) =
            snapping.as_ref().map_or_else(
                || (Vec::new(), Vec::new(), None),
                |snapping| {
                    let mut snapping = snapping.borrow_mut();
                    // The compatibility-only query must not create independent
                    // boundary diagnostics or make an unused policy active.
                    let diagnostics = snapping.diagnostics().to_vec();
                    let zero_image = snapping_threshold
                        .as_ref()
                        .filter(|value| value.is_zero())
                        .map(|_| {
                            snapping.snap_scalar_product(Atom::Zero, "zero-distance compatibility")
                        });
                    (snapping.aliases().to_vec(), diagnostics, zero_image)
                },
            );
        if certified_planar_chart.is_some() && !ordinary {
            if leaves.iter().any(|leaf| leaf.kind != MasterLeafKind::FourDimensional || leaf.active.len() > 4) {
                return Err(ReductionError::Preparation("certified planar scalar reduction emitted an unexpected high-point master".into()));
            }
            preparation_notes.push("Certified planar generalized scalar: exact constant-one denominator identity, original invariants preserved, and IBP pinches with at most four active propagators".into());
        }
        if matches!(certified_invariant_chart, Some(CertifiedInvariantChart::CartesianTensor(_))) {
            if ordinary && leaves.iter().any(|leaf| leaf.kind != MasterLeafKind::FourDimensional) {
                return Err(ReductionError::Preparation(
                    "ordinary checked supplied-G tensor completion emitted an unexpected shifted master".into()));
            }
            preparation_notes.push("Checked supplied-G Cartesian tensor chart: original G in scalar/master arguments, original p in moments; exact full-jet compatibility is required before execution".into());
            if !ordinary {
                preparation_notes.push("Generalized checked Cartesian tensors retain genuine scalar IBP, including D6 pentagon leaves with their existing endpoint, convergence and propagated-error guards; requested algorithm is preserved in settings".into());
            }
        }
        let snapping_zero_image_required = snapping_zero_image.is_some()
            && ((n == 1 && (algorithm_used == Algorithm::Ibp || generalized_ibp_used)
                && (settings.rank > 0 || !ordinary))
                || crate::snapping::highpoint_legacy_domain(&settings));
        let mut snapping_compatibility = if crate::snapping::highpoint_legacy_domain(&settings) {
            preparation_notes.push("Highpoint legacy on-shell domain is conservative: every demanded selected snap must equal its original Cartesian polynomial in every Taylor component; geometry-changing snaps fail before masters".into());
            snapping.as_ref().map_or_else(Vec::new,
                |policy| policy.borrow().compatibility_queries().to_vec())
        } else { Vec::new() };
        snapping_compatibility.extend(coordinate_constraints.into_iter().map(|constraint|
            crate::snapping::SnapCompatibilityQuery {
                kind: crate::snapping::CartesianConstraintKind::CoordinateIdentity,
                output: constraint.owner,
                raw: constraint.expression,
                label: constraint.label,
                choices: vec![(constraint.active, crate::snapping::SnapTarget::Zero)],
            }));
        if !snapping_aliases.is_empty() {
            let mut ordered = snapping_aliases.clone();
            ordered.append(&mut aliases);
            aliases = ordered;
        }
        apply_final_uv_power_counting(&settings, &basis, &mut moments)?;
        for (moment, monomial) in moments.iter_mut().zip(&basis) {
            *moment = convert_convention(moment, *monomial, &settings);
        }
        let master_arguments = leaves
            .iter()
            .map(|leaf| {
                let active = &leaf.active;
                let mut args = Vec::new();
                let mut distance =
                    |i: usize, j: usize| args.push(distances[active[i]][active[j]].clone());
                match (leaf.kind, active.len()) {
                    (MasterLeafKind::FourDimensional, 1) => {}
                    (MasterLeafKind::FourDimensional, 2) => distance(0, 1),
                    (MasterLeafKind::FourDimensional, 3) => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(0, 2);
                    }
                    (MasterLeafKind::FourDimensional, 4) => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(2, 3);
                        distance(0, 3);
                        distance(0, 2);
                        distance(1, 3);
                    }
                    (MasterLeafKind::SixDimensionalPentagon, 5) => {
                        for i in 0..5 {
                            for j in i + 1..5 {
                                distance(i, j);
                            }
                        }
                    }
                    _ => unreachable!("scalar compiler emitted an inconsistent master kind"),
                }
                args.extend(active.iter().map(|&i| masses[i].clone()));
                args.push(mu.pow(2));
                args
            })
            .collect();
        let fallback = if certified_affine_chart.is_none() && !special_kinematics
            && settings.algorithm == Algorithm::Hybrid
            && algorithm_used == Algorithm::PassarinoVeltman
        {
            let mut fallback_settings = settings.clone();
            fallback_settings.algorithm = Algorithm::Ibp;
            match Self::build(
                fallback_settings,
                None,
                None,
                analytic_jet_order,
                certified_invariant_chart,
                certified_planar_chart,
                certified_planar_mask,
            ) {
                Ok(plan) if plan.algorithm_used == Algorithm::Ibp => Some(Box::new(plan)),
                Ok(_) => {
                    preparation_notes.push("Independent IBP fallback is unavailable for this topology; its tensor completion also uses PV".into());
                    None
                }
                Err(error) => {
                    preparation_notes.push(format!("IBP fallback preparation failed: {error}"));
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            execution_cache: Default::default(),
            variant_cache: Default::default(),
            settings,
            basis,
            parameters,
            external_primal_seeds: Vec::new(),
            layout,
            distances,
            raw_distances,
            moments,
            leaves,
            master_arguments,
            aliases,
            snapping_aliases,
            snapping_diagnostics,
            snapping_threshold,
            snapping_zero_image,
            snapping_zero_image_required,
            snapping_compatibility,
            fallback,
            algorithm_used,
            preparation_notes,
            special_kinematics,
            analytic_jet_order,
            bubble_limit,
            zero_invariants,
            certified_invariant_chart,
            certified_planar_chart,
            certified_planar_mask,
            certified_affine_chart,
            thin_six_child_rank: None,
            thin_seven_child_rank: None,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    pub fn input_layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }
    pub fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub fn master_count(&self) -> usize {
        self.leaves.len()
    }
    pub fn algorithm_used(&self) -> Algorithm {
        self.algorithm_used
    }
    /// An IBP invariant profile cannot replace the distinct direct-dot Gram
    /// matrix selected by legacy PV snapping.
    pub(crate) fn zero_profile_dispatch_allowed(&self) -> bool {
        !self.special_kinematics
            && self.certified_affine_chart.is_none()
            // Rounded invariant evaluations cannot establish new exact-zero
            // constraints beyond the certified full-Taylor Gram relations.
            && self.certified_invariant_chart.is_none()
            && self.certified_planar_chart.is_none()
            && !(self.snapping_threshold.is_some()
                && self.algorithm_used == Algorithm::PassarinoVeltman)
    }
    /// Only ordinary requested PV/Hybrid plans may own the outer planar retry.
    /// The condition is retained when a plan is specialized for a Dual shape.
    pub(crate) fn automatic_planar_settings(&self) -> Option<ReductionSettings> {
        let settings = &self.settings;
        if self.special_kinematics
            || self.certified_invariant_chart.is_some()
            || self.certified_planar_chart.is_some()
            || settings.propagators != 5
            || settings.rank > 7
            || settings.powers != [1; 5]
            || settings.dimension_shift != 0
            || !matches!(
                settings.algorithm,
                Algorithm::PassarinoVeltman | Algorithm::Hybrid
            )
            || !(settings.invariants == InvariantInput::FromShifts
                || crate::certified_planar::supplied_six_point_child_settings(settings))
            || settings.input_policy != InputPolicy::Preserve
        {
            return None;
        }
        let mut settings = settings.clone();
        settings.algorithm = Algorithm::PassarinoVeltman;
        Some(settings)
    }

    pub(crate) fn automatic_affine_settings(&self) -> Option<ReductionSettings> {
        (!self.special_kinematics && self.certified_affine_chart.is_none()
            && self.certified_invariant_chart.is_none() && self.certified_planar_chart.is_none()
            && crate::certified_affine::settings_supported(&self.settings)).then(||self.settings.clone())
    }

    pub fn preparation_notes(&self) -> &[String] {
        &self.preparation_notes
    }
    /// Shared Symbolica recurrence nodes referenced by `expressions()`.
    pub fn aliases(&self) -> &[(Atom, Atom)] {
        &self.aliases
    }
    pub fn expressions(&self) -> &[[Atom; 4]] {
        &self.moments
    }

    /// Locally guarded comparison probes for the effective legacy policy.
    /// Runtime evaluation must also respect each probe owner's branch demand.
    pub fn snapping_diagnostics(&self) -> &[SnapDiagnostic] {
        &self.snapping_diagnostics
    }

    /// Exact threshold expression, absent when the policy has no effect.
    pub fn snapping_threshold(&self) -> Option<&Atom> {
        self.snapping_threshold.as_ref()
    }
}

fn compile_pv(
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    distances: &[Vec<Atom>],
    analytic_jet_order: u16,
    snapping: Option<Rc<RefCell<LegacySnapping>>>,
    certified_planar_chart: Option<usize>,
    certified_planar_mask: u8,
    coordinate_constraints: Option<&mut Vec<crate::pv::CoordinateConstraint>>,
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
    let mut compiler =
        PvCompiler::with_distances(shifts.to_vec(), masses.to_vec(), distances.to_vec(), true)
            .map_err(ReductionError::Preparation)?;
    compiler
        .set_analytic_jet_order(analytic_jet_order)
        .map_err(ReductionError::Preparation)?;
    if coordinate_constraints.is_some() {
        compiler.set_checked_coordinate_identities().map_err(ReductionError::Preparation)?;
    }
    if let Some(omitted) = certified_planar_chart {
        compiler
            .set_certified_planar_chart(omitted)
            .map_err(ReductionError::Preparation)?;
        compiler.set_certified_planar_mask(certified_planar_mask).map_err(ReductionError::Preparation)?;
    }
    if let Some(snapping) = snapping {
        compiler
            .set_gram_scalar_product_transform(move |raw| {
                snapping
                    .borrow_mut()
                    .snap_scalar_product(raw, "PV Gram scalar product")
            })
            .map_err(ReductionError::Preparation)?;
    }
    let moments = basis
        .iter()
        .map(|m| compiler.compile_moment(m.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ReductionError::Preparation)?;
    if let Some(constraints) = coordinate_constraints {
        constraints.extend_from_slice(compiler.coordinate_constraints());
    }
    Ok((
        moments,
        compiler.leaves().to_vec(),
        compiler.aliases().to_vec(),
    ))
}

fn compile_ibp(
    settings: &ReductionSettings,
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    supplied_distances: &[Vec<Atom>],
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<&ZeroInvariantProfile>,
    certified_basis: Option<[usize; 4]>,
    certified_planar_chart: Option<usize>,
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
    compile_scalar_tensors(
        settings,
        basis,
        shifts,
        masses,
        supplied_distances,
        bubble_limit,
        zero_invariants,
        certified_basis,
        certified_planar_chart,
        None,
    )
    .map(|(moments, leaves, aliases, _)| (moments, leaves, aliases))
}

#[derive(Default)]
struct ScalarMethodUsage {
    pv: usize,
    ibp: usize,
}

fn compile_generalized_pv(
    settings: &ReductionSettings,
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    distances: &[Vec<Atom>],
    analytic_jet_order: u16,
    snapping: Option<Rc<RefCell<LegacySnapping>>>,
    certified_planar_chart: Option<usize>,
) -> Result<
    (
        Vec<[Atom; 4]>,
        Vec<MasterLeaf>,
        Vec<(Atom, Atom)>,
        ScalarMethodUsage,
    ),
    ReductionError,
> {
    let pv = if shifts.len() <= 4 {
        let mut pv =
            PvCompiler::with_distances(shifts.to_vec(), masses.to_vec(), distances.to_vec(), true)
                .map_err(ReductionError::Preparation)?;
        pv.set_analytic_jet_order(analytic_jet_order)
            .map_err(ReductionError::Preparation)?;
        if let Some(snapping) = snapping {
            pv.set_gram_scalar_product_transform(move |raw| {
                snapping
                    .borrow_mut()
                    .snap_scalar_product(raw, "PV Gram scalar product")
            })
            .map_err(ReductionError::Preparation)?;
        }
        Some(pv)
    } else {
        None
    };
    compile_scalar_tensors(
        settings, basis, shifts, masses, distances, None, None, None,
        certified_planar_chart, pv,
    )
}

fn compile_scalar_tensors(
    settings: &ReductionSettings,
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    supplied_distances: &[Vec<Atom>],
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<&ZeroInvariantProfile>,
    certified_basis: Option<[usize; 4]>,
    certified_planar_chart: Option<usize>,
    mut pv: Option<PvCompiler>,
) -> Result<
    (
        Vec<[Atom; 4]>,
        Vec<MasterLeaf>,
        Vec<(Atom, Atom)>,
        ScalarMethodUsage,
    ),
    ReductionError,
> {
    // Independent squared-distance symbols keep exact matrix elimination small.
    // Symbolica shares their definitions across the recurrence graph.
    let n = shifts.len();
    let mut aliases = Vec::new();
    let mut distances = vec![vec![Atom::new(); n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let s = variable(&format!("s{i}_{j}"));
            distances[i][j] = s.clone();
            distances[j][i] = s.clone();
            aliases.push((s, supplied_distances[i][j].clone()));
        }
    }
    let r = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| &distances[i][j] - &masses[i] - &masses[j])
                .collect()
        })
        .collect();
    let mut compiler = match bubble_limit {
        Some((order, equal_masses)) => ScalarCompiler::lightlike_bubble(r, order, equal_masses),
        None => match zero_invariants {
            Some(profile) => {
                ScalarCompiler::with_zero_invariants(r, &profile.masses, &profile.distances)
            }
            None => ScalarCompiler::new(r),
        },
    }
    .map_err(ReductionError::Preparation)?;
    if let Some(omitted) = certified_planar_chart {
        // Certification guarantees that the omitted coordinate row depends on
        // the retained three, with an invertible constant-one chart. This only
        // selects relation coordinates: R and every numerical/master argument
        // still use the full original Cartesian shifts and squared distances.
        // Do not project the supplied invariants or change tensor projections.
        let coordinates = shifts.iter().map(|shift| {
            std::array::from_fn(|mu| if mu == omitted { Atom::new() } else { shift[mu].clone() })
        }).collect();
        compiler
            .set_certified_planar_coordinates(coordinates)
            .map_err(ReductionError::Preparation)?;
    } else if let Some(basis) = certified_basis {
        // The opt-in scalar wrapper certifies that these four Gram covectors
        // preserve every external linear dependency in every Taylor component.
        // They are coordinates on the span, not reconstructed Cartesian shifts;
        // original squared distances and masses remain authoritative in R.
        let coordinates =
            certified_covector_aliases(basis, &distances, supplied_distances, &mut aliases);
        compiler
            .set_certified_linear_coordinates(coordinates)
            .map_err(ReductionError::Preparation)?;
    } else if settings.invariants == InvariantInput::FromShifts
        && (settings.input_policy == InputPolicy::Preserve
            || crate::snapping::highpoint_legacy_domain(settings))
    {
        compiler
            .set_routing(shifts.to_vec())
            .map_err(ReductionError::Preparation)?;
    }
    compiler
        .enable_aliases()
        .map_err(ReductionError::Preparation)?;
    compiler
        .enable_physical_poles()
        .map_err(ReductionError::Preparation)?;
    compiler
        .enable_gram_regular_bubbles()
        .map_err(ReductionError::Preparation)?;
    if certified_planar_chart.is_none()
        && (settings.dimension_shift != 0 || settings.powers.iter().any(|&power| power != 1))
    {
        // Ordinary four-dimensional high-point tensors retain the existing
        // PV completion. Other generalized requests may require an independent
        // shifted master, whose backend enforces a finite parameter domain.
        // Checked planar scalar charts must instead complete by exact pinches;
        // never admit a genuine pentagon leaf as a hidden chart fallback.
        compiler
            .enable_six_dimensional_pentagon()
            .map_err(ReductionError::Preparation)?;
    }
    let t: [Atom; 4] = std::array::from_fn(|i| variable(&format!("aux{i}")));
    let metric = t[0].pow(2) - t[1].pow(2) - t[2].pow(2) - t[3].pow(2);
    let projections: Vec<Atom> = shifts
        .iter()
        .map(|p| t.iter().zip(p).map(|(a, b)| a * b).sum())
        .collect();
    let mut by_exponent: HashMap<[u16; 4], [Atom; 4]> = HashMap::new();
    let mut usage = ScalarMethodUsage::default();
    for rank in 0..=settings.rank {
        let mut generating: [Atom; 4] = std::array::from_fn(|_| Atom::new());
        for pairs in 0..=rank / 2 {
            let scalar_h = settings
                .dimension_shift
                .checked_add((rank - pairs) as i16)
                .ok_or_else(|| ReductionError::InvalidInput("dimension shift overflow".into()))?;
            let front = Atom::num(Integer::factorial(u32::from(rank)))
                / Atom::num(Integer::factorial(u32::from(pairs)))
                * Atom::num(-4).pow(-i64::from(pairs))
                * metric.pow(i64::from(pairs));
            for counts in compositions(rank - 2 * pairs, n) {
                let mut powers = settings.powers.clone();
                let mut coefficient = front.clone();
                for i in 0..n {
                    powers[i] = powers[i].checked_add(counts[i]).ok_or_else(|| {
                        ReductionError::InvalidInput("propagator power overflow".into())
                    })?;
                    if counts[i] != 0 {
                        coefficient = coefficient * projections[i].pow(i64::from(counts[i]));
                    }
                    coefficient = coefficient
                        * Atom::num(Integer::binom(
                            i64::from(settings.powers[i]) + i64::from(counts[i]) - 1,
                            i64::from(counts[i]),
                        ));
                }
                let request = ScalarRequest {
                    powers,
                    dimension_shift: scalar_h,
                };
                let pv_value = pv
                    .as_mut()
                    .map(|pv| pv.compile_generalized_scalar(&request))
                    .transpose()
                    .map_err(ReductionError::Preparation)?
                    .flatten();
                let value = if let Some(value) = pv_value {
                    usage.pv += 1;
                    value
                } else {
                    usage.ibp += 1;
                    compiler
                        .compile(request)
                        .map_err(ReductionError::Preparation)?
                };
                for pole in 0..4 {
                    generating[pole] = &generating[pole] + &coefficient * &value[pole];
                }
            }
        }
        // Symbolica performs the polynomial expansion and coefficient collection.
        for (pole, expression) in generating.into_iter().enumerate() {
            let field = AtomField {
                statistical_zero_test: false,
                cancel_check_on_division: true,
                custom_normalization: None,
            };
            let polynomial = expression
                .expand()
                .to_polynomial_in_vars_with_field::<u16>(&t, &field);
            for term in &polynomial {
                let exponents: [u16; 4] = term.exponents.try_into().map_err(|_| {
                    ReductionError::Preparation("unexpected generating polynomial basis".into())
                })?;
                let entry = by_exponent
                    .entry(exponents)
                    .or_insert_with(|| std::array::from_fn(|_| Atom::new()));
                let weight = multinomial(exponents);
                entry[pole] = term.coefficient / weight;
            }
        }
    }
    let moments = basis
        .iter()
        .map(|m| {
            by_exponent
                .remove(&m.0)
                .unwrap_or_else(|| std::array::from_fn(|_| Atom::new()))
        })
        .collect();
    aliases.extend_from_slice(compiler.aliases());
    if let Some(pv) = pv {
        aliases.extend_from_slice(pv.aliases());
        for leaf in pv.leaves() {
            if !compiler
                .leaves
                .iter()
                .any(|existing| existing.kind == leaf.kind && existing.active == leaf.active)
            {
                compiler.leaves.push(leaf.clone());
            }
        }
    }
    Ok((moments, compiler.leaves, aliases, usage))
}

/// Share the certified Gram covectors without expanding their input polynomials
/// inside every exact matrix operation. Symmetric entries share one handle.
/// These are coordinates on the certified span; physical R remains unchanged.
fn certified_covector_aliases(
    basis: [usize; 4],
    distances: &[Vec<Atom>],
    supplied_distances: &[Vec<Atom>],
    aliases: &mut Vec<(Atom, Atom)>,
) -> Vec<[Atom; 4]> {
    let mut entries = HashMap::new();
    (0..distances.len())
        .map(|i| {
            std::array::from_fn(|axis| {
                let anchor = basis[axis];
                if i == 0 {
                    return Atom::new();
                }
                if i == anchor {
                    return distances[0][i].clone();
                }
                let (first, second) = (anchor.min(i), anchor.max(i));
                entries
                    .entry((first, second))
                    .or_insert_with(|| {
                        let handle = variable(&format!("cert_k{first}_{second}"));
                        // Expand once in the original G inputs: this cancels
                        // repeated diagonal terms before numerical evaluation.
                        // Combining evaluated squared-distance aliases instead
                        // could introduce avoidable floating-point cancellation.
                        let body = ((&supplied_distances[0][anchor] + &supplied_distances[0][i]
                            - &supplied_distances[anchor][i])
                            / Atom::num(2))
                        .expand();
                        aliases.push((handle.clone(), body));
                        handle
                    })
                    .clone()
            })
        })
        .collect()
}

fn multinomial(exponents: [u16; 4]) -> Atom {
    let rank: u32 = exponents.iter().map(|&e| u32::from(e)).sum();
    exponents
        .into_iter()
        .fold(Atom::num(Integer::factorial(rank)), |value, e| {
            value / Atom::num(Integer::factorial(u32::from(e)))
        })
}

/// The existing final-output UV power-counting rule. Intermediate child poles
/// remain available to their own recurrence and finite rational terms.
pub(crate) fn apply_final_uv_power_counting(settings:&ReductionSettings, basis:&[Monomial], moments:&mut [[Atom;4]]) -> Result<(),ReductionError> {
    assert_eq!(basis.len(), moments.len(), "complete final tensor basis");
        let base_uv_degree = 4 + 2 * i32::from(settings.dimension_shift)
            - 2 * settings
                .powers
                .iter()
                .map(|&power| i32::from(power))
                .sum::<i32>();
        for (moment, monomial) in moments.iter_mut().zip(basis) {
            // Only the requested output is UV convergent. Intermediate UV
            // poles remain available to the recurrence's finite rational terms.
            let rank = i32::try_from(monomial.rank()).map_err(|_| {
                ReductionError::Unsupported("tensor rank exceeds UV power-counting range".into())
            })?;
            if base_uv_degree + rank < 0 {
                moment[0] = Atom::Zero;
            }
        }
        Ok(())
}

/// Bind every imported Laurent slot to its own child's mass dimension.
pub(crate) fn thin_six_external_primal_seeds(child_basis:&[Monomial], values:&[Atom], amplitudes:&[Atom])->Vec<(Atom,Atom,i32)> {
    let child_slots = child_basis.len()*4;
    assert!(!child_basis.is_empty());
    assert_eq!(values.len(),6*child_slots);
    assert_eq!(amplitudes.len(),values.len());
    values.iter().zip(amplitudes).enumerate().map(|(index,(value,amplitude))|
        (value.clone(),amplitude.clone(),child_basis[(index%child_slots)/4].rank() as i32-6)).collect()
}

/// Structural basis dispatch only; Symbolica performs every product and sum.
pub(crate) fn assemble_thin_six_moments(basis:&[Monomial], child_basis:&[Monomial], weights:&[Vec<Atom>;5], values:&[Atom])->Vec<[Atom;4]> {
    let child_slots=child_basis.len()*4;
    assert_eq!(values.len(),6*child_slots);
    assert!(weights.iter().all(|column|column.len()==6));
    let child_indices:HashMap<_,_>=child_basis.iter().enumerate().map(|(index,monomial)|(monomial.0,index)).collect();
    assert_eq!(child_indices.len(),child_basis.len(),"unique imported child basis");
    basis.iter().map(|monomial| {
        let mut remainder=monomial.0;
        let column=match remainder.iter().position(|&power|power!=0) {
            Some(mu)=>{remainder[mu]-=1;mu+1},None=>0,
        };
        let child_index=child_indices[&remainder];
        std::array::from_fn(|pole| Atom::add_many(&(0..6).map(|subset|
            &weights[column][subset]*&values[subset*child_slots+child_index*4+pole]).collect::<Vec<_>>()))
    }).collect()
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi::{name}")))
}
pub(crate) fn squared_difference(a: &[Atom; 4], b: &[Atom; 4]) -> Atom {
    (&a[0] - &b[0]).pow(2)
        - (&a[1] - &b[1]).pow(2)
        - (&a[2] - &b[2]).pow(2)
        - (&a[3] - &b[3]).pow(2)
}

// Integer index tuples describing tensor structures, not polynomial arithmetic.
fn compositions(total: u16, count: usize) -> Vec<Vec<u16>> {
    if count == 1 {
        return vec![vec![total]];
    }
    let mut result = Vec::new();
    for first in 0..=total {
        for mut tail in compositions(total - first, count - 1) {
            tail.insert(0, first);
            result.push(tail);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convention, Workspace};
    use iregi_masters::Float;
    use symbolica::domains::{dual::HyperDual, float::Complex, rational::Rational};

    fn settings(n: usize, rank: u16) -> ReductionSettings {
        let mut settings = ReductionSettings::new(n, rank);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings
    }

    #[test]
    fn generalized_pv_dispatch_preserves_negative_shift_and_mixed_terms() {
        for (powers, shift, rank, expected_method) in [
            (vec![1, 2], 1, 0, Algorithm::PassarinoVeltman),
            (vec![2, 1], 1, 0, Algorithm::Ibp),
            (vec![1, 1], -1, 0, Algorithm::Ibp),
            (vec![1, 1], -1, 2, Algorithm::PassarinoVeltman),
        ] {
            let mut configuration = settings(2, rank);
            configuration.algorithm = Algorithm::PassarinoVeltman;
            configuration.powers = powers;
            configuration.dimension_shift = shift;
            let plan = ReductionPlan::new(configuration.clone()).unwrap();
            assert_eq!(plan.algorithm_used(), expected_method);
            assert!(
                plan.preparation_notes()
                    .iter()
                    .any(|note| note.contains("Generalized scalar-term dispatch"))
            );
            if shift < 0 && rank == 2 {
                assert!(
                    plan.preparation_notes()
                        .iter()
                        .any(|note| note.contains("1 canonical PV requests"))
                );
            }
            // Translation makes numerator terms raising the first denominator
            // nonzero, so a per-input eligibility shortcut would be exposed.
            let row = [1., 2., 0., 0., 1., 3., 0., 0., 4., 9., 3.];
            let input = crate::KinematicsBatch {
                rows: 1,
                components: 1,
                values: row.map(|value| Complex::new(value, 0.)).to_vec(),
            };
            let mut actual_workspace = Workspace::new(&plan, 0.).unwrap();
            let actual = actual_workspace.reduce_batch(&input).unwrap();
            configuration.algorithm = Algorithm::Ibp;
            let expected_plan = ReductionPlan::new(configuration).unwrap();
            let mut expected_workspace = Workspace::new(&expected_plan, 0.).unwrap();
            let expected = expected_workspace.reduce_batch(&input).unwrap();
            for (left, right) in actual
                .lane(0)
                .unwrap()
                .iter()
                .zip(expected.lane(0).unwrap())
            {
                let error = (left.re - right.re).hypot(left.im - right.im);
                assert!(
                    error <= 1e-10 * (1. + right.re.hypot(right.im)),
                    "{left:?} versus {right:?}"
                );
            }
            let unique: std::collections::HashSet<_> = plan
                .leaves
                .iter()
                .map(|leaf| (leaf.kind, leaf.active.clone()))
                .collect();
            assert_eq!(unique.len(), plan.leaves.len());
        }
    }

    #[test]
    fn generalized_pv_native_normalization_retains_dimension_and_uv_residue() {
        for convention in [Convention::OneLoop, Convention::IregiNative] {
            let mut configuration = settings(1, 0);
            configuration.algorithm = Algorithm::PassarinoVeltman;
            configuration.dimension_shift = 1;
            configuration.convention = convention;
            let plan = ReductionPlan::new(configuration).unwrap();
            let uv = plan
                .aliases
                .iter()
                .rev()
                .fold(plan.moments[0][0].clone(), |value, (handle, body)| {
                    value.replace(handle.clone()).with(body.clone())
                });
            let mass_squared = &plan.parameters[plan.layout.masses.start];
            let expected = -mass_squared.pow(2) / Atom::num(2);
            let expected = if convention == Convention::IregiNative {
                expected * symbolica::parse!("𝑖") * Atom::var(symbolica::atom::Symbol::PI).pow(3)
            } else {
                expected
            };
            assert!((uv - expected).together().is_zero());
        }
    }

    #[test]
    fn generalized_pv_highpoint_keeps_genuine_shifted_master_and_method_note() {
        let mut configuration = settings(5, 0);
        configuration.algorithm = Algorithm::PassarinoVeltman;
        configuration.dimension_shift = 1;
        let plan = ReductionPlan::new(configuration).unwrap();
        assert_eq!(plan.algorithm_used(), Algorithm::Ibp);
        assert!(
            plan.leaves
                .iter()
                .any(|leaf| leaf.kind == MasterLeafKind::SixDimensionalPentagon)
        );
        assert!(
            plan.preparation_notes()
                .iter()
                .any(|note| note.contains("not qualified") && note.contains("D6"))
        );
    }

    #[test]
    fn generalized_pv_hybrid_falls_back_when_mapped_coefficient_exceeds_budget() {
        let mut configuration = settings(1, 0);
        configuration.algorithm = Algorithm::PassarinoVeltman;
        configuration.dimension_shift = 17;
        assert!(
            matches!(ReductionPlan::new(configuration.clone()), Err(ReductionError::Preparation(reason))
            if reason.contains("preparation budget"))
        );
        configuration.algorithm = Algorithm::Hybrid;
        let plan = ReductionPlan::new(configuration).unwrap();
        assert_eq!(plan.algorithm_used(), Algorithm::Ibp);
        assert!(
            plan.preparation_notes()
                .iter()
                .any(|note| note.contains("PV preparation failed; using IBP"))
        );
    }

    #[test]
    fn generalized_pv_mixed_tadpole_retains_legacy_zero_diagonal_guard() {
        let mut configuration = settings(1, 1);
        configuration.dimension_shift = 1;
        configuration.algorithm = Algorithm::PassarinoVeltman;
        configuration.input_policy = InputPolicy::LegacyOnShell {
            threshold: "0".into(),
        };
        let plan = ReductionPlan::new(configuration).unwrap();
        assert_eq!(plan.algorithm_used(), Algorithm::PassarinoVeltman);
        assert!(plan.snapping_zero_image_required);
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let batch = crate::KinematicsBatch {
            rows: 2,
            components: 1,
            values: [-4., 4.]
                .into_iter()
                .flat_map(|mass| [1., 0., 0., 0., mass, 3.].map(|value| Complex::new(value, 0.)))
                .collect(),
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert!(
            matches!(&result.statuses[0], crate::LaneStatus::InvalidInput(reason)
            if reason.contains("nonphysical Cayley-diagonal domain"))
        );
        assert_eq!(result.statuses[1], crate::LaneStatus::Success);
        assert_eq!(workspace.last_execution().master_rows, 1);
    }

    #[test]
    fn generalized_pv_original_generalized_fixtures_agree_in_all_modes() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../../validation/reference/fixtures.json"))
                .unwrap();
        let mut comparisons = 0;
        for case in fixtures["cases"].as_array().unwrap() {
            let input = &case["input"];
            if input["api"] != "general" {
                continue;
            }
            let propagators = input["propagators"].as_array().unwrap();
            let mut configuration =
                ReductionSettings::new(propagators.len(), input["rank"].as_u64().unwrap() as u16);
            configuration.dimension_shift = (input["dimension_shift"].as_i64().unwrap() / 2) as i16;
            configuration.powers = propagators
                .iter()
                .map(|p| p["power"].as_u64().unwrap() as u16)
                .collect();
            let base_dimension = 4 + 2 * i32::from(configuration.dimension_shift)
                - 2 * configuration
                    .powers
                    .iter()
                    .map(|&power| i32::from(power))
                    .sum::<i32>();
            let mut characteristic_scale = input["mu"].as_f64().unwrap();
            let mut values = Vec::new();
            for propagator in propagators {
                for value in propagator["shift"].as_array().unwrap() {
                    let value = value.as_f64().unwrap();
                    characteristic_scale = characteristic_scale.max(value.abs());
                    values.push(Complex::new(value, 0.));
                }
            }
            for propagator in propagators {
                let mass = propagator["mass_squared"].as_array().unwrap();
                let mass = Complex::new(mass[0].as_f64().unwrap(), mass[1].as_f64().unwrap());
                characteristic_scale = characteristic_scale.max(mass.re.hypot(mass.im).sqrt());
                values.push(mass);
            }
            values.push(Complex::new(input["mu"].as_f64().unwrap(), 0.));
            let batch = crate::KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            };
            for mode in [
                Algorithm::Ibp,
                Algorithm::PassarinoVeltman,
                Algorithm::Hybrid,
            ] {
                configuration.algorithm = mode;
                let plan = ReductionPlan::new(configuration.clone()).unwrap();
                let mut workspace = crate::PrecisionRescueWorkspace::new(
                    &plan,
                    0.,
                    crate::RescueSettings::default(),
                )
                .unwrap();
                let result = workspace.reduce_batch(&batch).unwrap();
                let actual = result
                    .lane(0)
                    .unwrap_or_else(|error| panic!("{} {mode:?}: {error:?}", input["id"]));
                let expected = case["expected"]["moments"].as_array().unwrap();
                assert_eq!(actual.len(), expected.len() * 4);
                for (index, expected) in expected.iter().enumerate() {
                    let natural = characteristic_scale
                        .powi(base_dimension + plan.basis()[index].rank() as i32);
                    for slot in 0..4 {
                        let re = expected[slot][0].as_f64().unwrap();
                        let im = expected[slot][1].as_f64().unwrap();
                        let value = &actual[4 * index + slot];
                        assert!(
                            (value.re - re).hypot(value.im - im)
                                <= 1e-10 * (natural + re.hypot(im)),
                            "{} {mode:?} moment {index}, slot {slot}: {value:?} expected ({re},{im})",
                            input["id"]
                        );
                    }
                }
                comparisons += 1;
            }
        }
        assert_eq!(comparisons, 21);
    }

    #[test]
    fn generalized_pv_second_order_dual_batches_match_ibp() {
        let mut configuration = settings(2, 1);
        configuration.powers = vec![1, 2];
        configuration.dimension_shift = 2;
        configuration.algorithm = Algorithm::PassarinoVeltman;
        let pv = ReductionPlan::new(configuration.clone()).unwrap();
        configuration.algorithm = Algorithm::Ibp;
        let ibp = ReductionPlan::new(configuration).unwrap();
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut pv = Workspace::new_dual(&pv, 0., shape.clone()).unwrap();
        let mut ibp = Workspace::new_dual(&ibp, 0., shape).unwrap();
        let values = (0..3)
            .flat_map(|lane| {
                [1., 2., 0., 0., 1., 3., 0., 0., 4. + lane as f64, 9., 3.]
                    .into_iter()
                    .enumerate()
                    .flat_map(|(argument, value)| {
                        [value, if argument == 8 { 1. } else { 0. }, 0.]
                            .map(|value| Complex::new(value, 0.))
                    })
            })
            .collect();
        let batch = crate::KinematicsBatch {
            rows: 3,
            components: 3,
            values,
        };
        let actual = pv.reduce_batch(&batch).unwrap();
        let expected = ibp.reduce_batch(&batch).unwrap();
        for lane in 0..3 {
            for (left, right) in actual
                .lane(lane)
                .unwrap()
                .iter()
                .zip(expected.lane(lane).unwrap())
            {
                assert!(
                    (left.re - right.re).hypot(left.im - right.im)
                        <= 1e-10 * (1. + right.re.hypot(right.im)),
                    "lane {lane}: {left:?} versus {right:?}"
                );
            }
        }
        assert!(pv.last_execution().master_rows >= 3);
    }

    #[test]
    fn certified_covector_aliases_preserve_exact_inputs_and_mixed_dual_components() {
        use symbolica::{
            domains::float::SingleFloat,
            evaluate::{Dualizer, ExpressionEvaluator},
        };

        for (n, basis) in [(6, [1, 2, 3, 4]), (6, [2, 3, 4, 5]), (7, [2, 4, 5, 6])] {
            let mut gram = vec![vec![Atom::new(); n]; n];
            let mut parameters = Vec::new();
            for i in 0..n {
                for j in i..n {
                    let value = variable(&format!("covector_test_g{i}_{j}"));
                    gram[i][j] = value.clone();
                    gram[j][i] = value.clone();
                    parameters.push(value);
                }
            }
            let mut distances = vec![vec![Atom::new(); n]; n];
            let mut raw = distances.clone();
            let mut aliases = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    let value = variable(&format!("covector_test_s{i}_{j}"));
                    let body = &gram[i][i] + &gram[j][j] - Atom::num(2) * &gram[i][j];
                    distances[i][j] = value.clone();
                    distances[j][i] = value.clone();
                    raw[i][j] = body.clone();
                    raw[j][i] = body.clone();
                    aliases.push((value, body));
                }
            }
            let coordinates = certified_covector_aliases(basis, &distances, &raw, &mut aliases);
            assert!(coordinates[0].iter().all(AtomCore::is_zero));
            for (axis, &anchor) in basis.iter().enumerate() {
                assert_eq!(coordinates[anchor][axis], distances[0][anchor]);
                for (other_axis, &other) in basis.iter().enumerate() {
                    assert_eq!(coordinates[other][axis], coordinates[anchor][other_axis]);
                }
            }
            let mut expressions = Vec::new();
            for (i, coordinate) in coordinates.iter().enumerate() {
                for (axis, &anchor) in basis.iter().enumerate() {
                    let expected = &gram[anchor][i] - &gram[0][anchor] - &gram[0][i] + &gram[0][0];
                    let expanded = aliases
                        .iter()
                        .rev()
                        .fold(coordinate[axis].clone(), |value, (handle, body)| {
                            value.replace(handle.clone()).with(body.clone())
                        });
                    assert!((expanded - &expected).together().is_zero());
                    expressions.extend([coordinate[axis].clone(), expected]);
                }
            }
            // This exercises alias propagation through Symbolica's Dualizer,
            // including reordered, mixed and second-order Taylor components.
            let shape = vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![0, 2],
                vec![2, 0],
            ];
            let exact: ExpressionEvaluator<Complex<Rational>> =
                Atom::evaluator_multiple(&expressions, &parameters)
                    .add_aliases(aliases)
                    .unwrap()
                    .direct_translation(true)
                    .horner_iterations(0)
                    .cores(1)
                    .build()
                    .unwrap();
            let prototype = Float::new(256);
            let convert = |value: &Complex<Rational>| {
                Complex::new(
                    prototype.from_rational(&value.re),
                    prototype.from_rational(&value.im),
                )
            };
            let mut evaluator = exact
                .vectorize(&Dualizer::new(
                    HyperDual::<Complex<Rational>>::new(shape.clone()),
                    Vec::new(),
                ))
                .unwrap()
                .map_coeff_with_prec(&convert, 256);
            let inputs = (0..parameters.len() * shape.len())
                .map(|i| {
                    let argument = (i / shape.len()) as i64;
                    let component = (i % shape.len()) as i64;
                    Complex::new(
                        Rational::from(
                            (argument + 3).pow(2) * (component + 1).pow(3)
                                + (argument + 1) * (component + 5),
                        ),
                        Rational::from(
                            (argument + 2).pow(3) * (component + 2).pow(2)
                                + (argument + 7) * (component + 3),
                        ),
                    )
                })
                .map(|value| convert(&value))
                .collect::<Vec<_>>();
            // These integer jets and the linear half-integer coefficients are
            // represented exactly at 256 bits; equality needs no tolerance.
            let mut outputs = vec![
                convert(&Complex::new(Rational::from(0), Rational::from(0)));
                expressions.len() * shape.len()
            ];
            evaluator.evaluate(&inputs, &mut outputs);
            let mut distinct_mixed_components = false;
            for pair in outputs.chunks_exact(2 * shape.len()) {
                assert_eq!(&pair[..shape.len()], &pair[shape.len()..]);
                distinct_mixed_components |= pair[3] != pair[0] && pair[4] != pair[5];
            }
            assert!(distinct_mixed_components);
        }
    }

    #[test]
    fn certified_scalar_chart_rejects_unsupported_domains_and_invalid_bases() {
        let mut configuration = settings(6, 0);
        configuration.invariants = InvariantInput::SuppliedScalarProducts;
        assert!(matches!(
            ReductionPlan::new(configuration.clone()),
            Err(ReductionError::Unsupported(_))
        ));
        for chart in [[0, 1, 2, 3], [1, 1, 3, 4], [1, 2, 3, 6]] {
            assert!(matches!(
                ReductionPlan::new_certified_scalar(configuration.clone(), chart),
                Err(ReductionError::InvalidInput(_))
            ));
        }
        let mut non_scalar = configuration.clone();
        non_scalar.rank = 1;
        let mut coordinate_input = configuration.clone();
        coordinate_input.invariants = InvariantInput::FromShifts;
        let mut pv = configuration.clone();
        pv.algorithm = Algorithm::PassarinoVeltman;
        let mut hybrid = configuration;
        hybrid.algorithm = Algorithm::Hybrid;
        let mut five = settings(5, 0);
        five.invariants = InvariantInput::SuppliedScalarProducts;
        let mut eight = settings(8, 0);
        eight.invariants = InvariantInput::SuppliedScalarProducts;
        for invalid in [non_scalar, coordinate_input, pv, hybrid, five, eight] {
            assert!(matches!(
                ReductionPlan::new_certified_scalar(invalid, [1, 2, 3, 4]),
                Err(ReductionError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn nontrivial_dual_specializations_share_exact_work_across_precisions() {
        let plan = ReductionPlan::new(settings(1, 0)).unwrap();
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let binary = Workspace::new_dual(&plan, 0., shape.clone()).unwrap();
        let specialized = plan.with_analytic_jet_order(1).unwrap();
        let first = specialized
            .execution_cache
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        let precise = Workspace::new_dual(&plan, Float::new(256), shape).unwrap();
        let again = plan.with_analytic_jet_order(1).unwrap();
        let second = again
            .execution_cache
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        assert!(Arc::ptr_eq(&specialized, &again));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(binary.binary_precision(), 53);
        assert_eq!(precise.binary_precision(), 256);

        // Same total degree, different Taylor-component order: the tensor
        // plan is shared, while each exact flattened program remains distinct.
        let first_shape = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
        let second_shape = vec![vec![0, 0], vec![0, 1], vec![1, 0]];
        let first = Workspace::new_dual(
            &plan,
            0.,
            HyperDual::<Complex<Rational>>::new(first_shape.clone()),
        )
        .unwrap();
        let second = Workspace::new_dual(
            &plan,
            0.,
            HyperDual::<Complex<Rational>>::new(second_shape.clone()),
        )
        .unwrap();
        assert_eq!(first.dual_shape().unwrap(), first_shape);
        assert_eq!(second.dual_shape().unwrap(), second_shape);
        let cache = specialized.execution_cache.lock().unwrap();
        assert_eq!(cache.len(), 3);
        let first = cache[&(Some(first_shape), crate::MasterExecutionPolicy::Default)]
            .get().unwrap().as_ref().unwrap();
        let second = cache[&(Some(second_shape), crate::MasterExecutionPolicy::Default)]
            .get().unwrap().as_ref().unwrap();
        assert!(!Arc::ptr_eq(first, second));
    }

    #[test]
    fn canonical_profiles_are_shared_and_variant_caches_do_not_form_cycles() {
        let weak = {
            let plan = ReductionPlan::new(settings(2, 0)).unwrap();
            let first = plan
                .with_zero_invariants(ZeroInvariantProfile {
                    masses: vec![true, true],
                    distances: vec![(1, 0), (0, 1)],
                })
                .unwrap();
            let second = plan
                .with_zero_invariants(ZeroInvariantProfile {
                    masses: vec![true, true],
                    distances: vec![(0, 1)],
                })
                .unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            let _binary = Workspace::new(&first, 0.).unwrap();
            let _precise = Workspace::new(&second, Float::new(256)).unwrap();
            assert_eq!(first.execution_cache.lock().unwrap().len(), 1);
            Arc::downgrade(&first)
        };
        assert!(weak.upgrade().is_none());
    }

    fn assert_concurrent_variant_lookup(plan: Arc<ReductionPlan>) {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let plan = plan.clone();
                std::thread::spawn(move || plan.with_analytic_jet_order(2).unwrap())
            })
            .collect();
        let variants: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(
            variants
                .iter()
                .all(|variant| Arc::ptr_eq(variant, &variants[0]))
        );
        let weak = Arc::downgrade(&variants[0]);
        drop(variants);
        drop(plan);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn warm_specializations_support_concurrent_cache_lookup() {
        let plan = Arc::new(ReductionPlan::new(settings(1, 0)).unwrap());
        // Initialize Symbolica's exact graph on the current licensed thread.
        // Worker threads receive only the completed Rust cache entry, so they
        // cannot enter any Symbolica construction or Atom-cloning code.
        let variant = plan.with_analytic_jet_order(2).unwrap();
        let entry = plan
            .variant_cache
            .lock()
            .unwrap()
            .get(&PlanVariant {
                analytic_jet_order: 2,
                bubble_equal_masses: None,
                zero_invariants: None,
                certified_planar_mask: 0, certified_affine_chart: None,
                certified_invariant_chart: None,
                thin_six_child_rank: None,
                thin_seven_child_rank: None,
            })
            .unwrap()
            .clone();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let entry = entry.clone();
                std::thread::spawn(move || entry.get().unwrap().as_ref().unwrap().clone())
            })
            .collect();
        for handle in handles {
            assert!(Arc::ptr_eq(&handle.join().unwrap(), &variant));
        }
        let weak = Arc::downgrade(&variant);
        drop(variant);
        drop(entry);
        drop(plan);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    #[ignore = "cold preparation on worker threads requires a multithread-enabled Symbolica license"]
    fn cold_specializations_are_initialized_once_across_threads() {
        assert_concurrent_variant_lookup(Arc::new(ReductionPlan::new(settings(1, 0)).unwrap()));
    }

    #[test]
    fn final_uv_power_counting_preserves_ir_and_dimension_shifted_residues() {
        eprintln!("UV fixture: dotted bubble in four dimensions");
        let mut dotted = settings(2, 0);
        dotted.powers = vec![2, 1];
        let convergent = ReductionPlan::new(dotted.clone()).unwrap();
        assert!(convergent.moments[0][0].is_zero());
        assert!(!convergent.moments[0][3].is_zero());
        eprintln!("UV fixture: dotted bubble in six dimensions");
        dotted.dimension_shift = 1;
        let shifted = ReductionPlan::new(dotted).unwrap();
        assert!(!shifted.moments[0][0].is_zero());

        eprintln!("UV fixture: scaleless bubble UV/IR cancellation");
        let bubble = ReductionPlan::new(settings(2, 0))
            .unwrap()
            .with_zero_invariants(ZeroInvariantProfile {
                masses: vec![true; 2],
                distances: vec![(0, 1)],
            })
            .unwrap();
        assert_eq!(bubble.moments[0][0], Atom::num(1));
        assert_eq!(bubble.moments[0][1], Atom::num(-1));

        eprintln!("UV fixture: massless on-shell triangle IR double pole");
        let triangle = ReductionPlan::new(settings(3, 0))
            .unwrap()
            .with_zero_invariants(ZeroInvariantProfile {
                masses: vec![true; 3],
                distances: vec![(0, 1), (0, 2)],
            })
            .unwrap();
        assert!(triangle.moments[0][0].is_zero());
        assert!(!triangle.moments[0][2].is_zero());
    }

    #[test]
    #[ignore = "expensive generic five-to-seven-point symbolic preparation; run with an explicit time budget"]
    fn final_uv_power_counting_high_point_symbolic_preparation() {
        for (n, rank) in [(5, 2), (6, 0), (7, 0)] {
            let started = std::time::Instant::now();
            eprintln!("UV high-point fixture: preparing {n} propagators, rank {rank}");
            let mut configuration = settings(n, rank);
            configuration.algorithm = Algorithm::PassarinoVeltman;
            let plan = ReductionPlan::new(configuration).unwrap();
            assert!(plan.moments.iter().all(|moment| moment[0].is_zero()));
            assert!(plan.moments.iter().any(|moment| !moment[3].is_zero()));
            eprintln!(
                "UV high-point fixture: {n} propagators, rank {rank} passed in {:?}",
                started.elapsed()
            );
        }
    }
}

#[cfg(test)]
mod planar_mask_variant_tests {
    use super::*;
    fn configuration()->ReductionSettings {let mut s=ReductionSettings::new(5,0);s.algorithm=Algorithm::PassarinoVeltman;s}
    #[test]
    fn mask_identity_survives_analytic_variant_rebuild_and_cache() {
        for mask in [31,32,128,255] {assert!(ReductionPlan::new_certified_planar_masked(configuration(),3,mask).is_err());}
        let masked=ReductionPlan::new_certified_planar_masked(configuration(),3,8).unwrap();
        let first=masked.with_analytic_jet_order(2).unwrap();let again=masked.with_analytic_jet_order(2).unwrap();
        assert_eq!(first.certified_planar_mask,8);assert_eq!(first.certified_planar_chart,Some(3));assert_eq!(first.analytic_jet_order,2);assert!(Arc::ptr_eq(&first,&again));
        assert!(masked.variant_cache.lock().unwrap().contains_key(&PlanVariant{analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,certified_planar_mask:8, certified_affine_chart: None,certified_invariant_chart:None,thin_six_child_rank:None,thin_seven_child_rank:None}));
        let unmasked=ReductionPlan::new_certified_planar(configuration(),3).unwrap().with_analytic_jet_order(2).unwrap();
        assert_eq!(unmasked.certified_planar_mask,0);assert!(!Arc::ptr_eq(&first,&unmasked));
        assert_ne!(first.expressions(),unmasked.expressions());
    }
}

#[cfg(test)]
mod generic_thin_variant_tests {
    use super::*;
    #[test]
    fn thin_variant_rebuild_retains_imported_topology_and_native_output_context() {
        let mut settings=ReductionSettings::new(6,0);
        settings.algorithm=Algorithm::PassarinoVeltman;
        let plan=ReductionPlan::new_thin_six(settings,0).unwrap();
        assert_eq!(plan.thin_six_child_rank,Some(0));
        assert_eq!(plan.parameters.len(),79);
        assert_eq!(plan.external_primal_seeds.len(),24);
        let rebuilt=plan.with_analytic_jet_order(2).unwrap();
        let cached=plan.with_analytic_jet_order(2).unwrap();
        assert!(Arc::ptr_eq(&rebuilt,&cached));
        assert_eq!(rebuilt.thin_six_child_rank,Some(0));
        assert_eq!(rebuilt.analytic_jet_order,2);
        assert_eq!(serde_json::to_value(rebuilt.settings()).unwrap(), serde_json::to_value(plan.settings()).unwrap());
        assert_eq!(rebuilt.basis(),plan.basis());
        assert_eq!(rebuilt.parameters,plan.parameters);
        assert_eq!(rebuilt.external_primal_seeds,plan.external_primal_seeds);
        assert_eq!(rebuilt.expressions(),plan.expressions());
        assert!(plan.variant_cache.lock().unwrap().contains_key(&PlanVariant {
            analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,
            certified_planar_mask:0, certified_affine_chart: None,certified_invariant_chart:None,thin_six_child_rank:Some(0),thin_seven_child_rank:None,
        }));
        assert!(plan.with_zero_invariants(ZeroInvariantProfile {masses:vec![false;6],distances:Vec::new()}).is_err());
        assert!(plan.variant(PlanVariant {
            analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,
            certified_planar_mask:0, certified_affine_chart: None,certified_invariant_chart:None,thin_six_child_rank:Some(1),thin_seven_child_rank:None,
        }).is_err());
        assert!(plan.variant(PlanVariant {
            analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,
            certified_planar_mask:8, certified_affine_chart: None,certified_invariant_chart:None,thin_six_child_rank:Some(0),thin_seven_child_rank:None,
        }).is_err());
    }
}

#[cfg(test)]
mod certified_planar_generalized_tests {
    use super::*;
    use crate::CertifiedPlanarReductionPlan;

    fn generalized(h: i16, dotted: Option<usize>, algorithm: Algorithm) -> ReductionSettings {
        let mut settings = ReductionSettings::new(5, 0);
        settings.dimension_shift = h;
        settings.algorithm = algorithm;
        if let Some(index) = dotted { settings.powers[index] = 2; }
        settings
    }

    #[test]
    fn generalized_planar_public_domain_is_narrow_and_scalar_only() {
        for algorithm in [Algorithm::Ibp, Algorithm::PassarinoVeltman, Algorithm::Hybrid] {
            for (h, dotted) in [(1, None), (0, Some(0)), (1, Some(4))] {
                let settings = generalized(h, dotted, algorithm);
                assert!(CertifiedPlanarReductionPlan::new(settings.clone()).is_ok());
                assert!(CertifiedPlanarReductionPlan::new_for_six_point_child(settings).is_err());
            }
        }
        for mut settings in [
            generalized(2, None, Algorithm::Ibp),
            generalized(-1, Some(0), Algorithm::Ibp),
            generalized(1, Some(0), Algorithm::Ibp),
            generalized(1, Some(0), Algorithm::Ibp),
            generalized(1, Some(0), Algorithm::Ibp),
        ].into_iter().enumerate().map(|(index, mut settings)| {
            match index {
                2 => settings.rank = 1,
                3 => settings.powers[1] = 2,
                4 => settings.powers[0] = 3,
                _ => (),
            }
            settings
        }) {
            assert!(CertifiedPlanarReductionPlan::new(settings.clone()).is_err());
            // The private constructor must enforce the same mathematical scope.
            assert!(ReductionPlan::new_certified_planar(settings.clone(), 0).is_err());
            settings.invariants = InvariantInput::SuppliedScalarProducts;
            assert!(CertifiedPlanarReductionPlan::new(settings).is_err());
        }
        assert!(ReductionPlan::new_certified_planar_masked(
            generalized(1, None, Algorithm::Ibp), 0, 1,
        ).is_err());
        assert!(ReductionPlan::new_certified_planar(
            generalized(1, None, Algorithm::Ibp), 4,
        ).is_err());
    }

    #[test]
    #[ignore = "new generic Symbolica chart preparation; run only with an explicit qualification budget"]
    fn generalized_pv_ibp_and_hybrid_all_retain_the_certified_scalar_route() {
        for algorithm in [Algorithm::Ibp, Algorithm::PassarinoVeltman, Algorithm::Hybrid] {
            for (h, dotted) in [(1, None), (0, Some(0)), (1, Some(4))] {
                let settings = generalized(h, dotted, algorithm);
                let plan = ReductionPlan::new_certified_planar(settings, 0).unwrap();
                assert_eq!(plan.algorithm_used(), Algorithm::Ibp);
                assert!(plan.automatic_planar_settings().is_none());
                assert!(!plan.zero_profile_dispatch_allowed());
                assert!(plan.fallback.is_none());
                assert!(plan.leaves.iter().all(|leaf|
                    leaf.kind == MasterLeafKind::FourDimensional && leaf.active.len() <= 4));
                assert!(!plan.leaves.is_empty());
                assert!(plan.preparation_notes().iter().any(|note|
                    note.contains("Certified planar generalized scalar")));
                let variant = plan.with_analytic_jet_order(1).unwrap();
                assert_eq!(variant.certified_planar_chart, Some(0));
                assert_eq!(variant.certified_planar_mask, 0);
                assert_eq!(variant.algorithm_used(), Algorithm::Ibp);
                assert!(variant.leaves.iter().all(|leaf|
                    leaf.kind == MasterLeafKind::FourDimensional && leaf.active.len() <= 4));
            }
        }
    }
}

#[cfg(test)]
mod checked_tensor_chart_context_tests {
    use super::*;

    #[test]
    fn scalar_tensor_and_unchecked_chart_modes_have_distinct_cache_identity() {
        let base = PlanVariant {
            analytic_jet_order: 2,
            bubble_equal_masses: None,
            zero_invariants: None,
            certified_planar_mask: 0, certified_affine_chart: None,
            thin_six_child_rank: None,
            thin_seven_child_rank: None,
            certified_invariant_chart: None,
        };
        let mut scalar = base.clone();
        scalar.certified_invariant_chart = Some(CertifiedInvariantChart::Scalar([1,2,3,4]));
        let mut tensor = base.clone();
        tensor.certified_invariant_chart = Some(CertifiedInvariantChart::CartesianTensor([1,2,3,4]));
        assert_ne!(base, scalar);
        assert_ne!(scalar, tensor);
        assert!(!scalar.certified_invariant_chart.unwrap().tensor_completion_allowed());
        assert!(tensor.certified_invariant_chart.unwrap().tensor_completion_allowed());
        let mut other_chart = tensor.clone();
        other_chart.certified_invariant_chart = Some(CertifiedInvariantChart::CartesianTensor([1,2,3,5]));
        assert_ne!(tensor, other_chart);
    }

    #[test]
    fn packed_supplied_tensor_variant_retains_original_input_and_imported_context() {
        let mut settings=ReductionSettings::new(6,0);
        settings.algorithm=Algorithm::PassarinoVeltman;
        settings.invariants=InvariantInput::SuppliedScalarProducts;
        settings.convention=crate::Convention::MadLoop;
        settings.tensor_weights=crate::TensorWeights::Multinomial;
        assert!(ReductionPlan::new(settings.clone()).is_err(),"unchecked public N6 G guard remains");
        let plan=ReductionPlan::new_certified_tensor(settings.clone(),[1,2,3,4]).unwrap();
        eprintln!("packed-G structural test: initial parent prepared");
        assert_eq!(plan.layout.scalar_products,Some(24..45));assert_eq!(plan.layout.masses,45..51);
        assert_eq!(plan.layout.scale,51);assert_eq!(plan.thin_six_child_rank,Some(0));
        assert_eq!(plan.parameter_count(),100);assert_eq!(plan.external_primal_seeds.len(),24);
        let variant=plan.with_analytic_jet_order(2).unwrap();let cached=plan.with_analytic_jet_order(2).unwrap();
        assert!(Arc::ptr_eq(&variant,&cached));assert_eq!(variant.parameters,plan.parameters);
        assert_eq!(variant.aliases,plan.aliases);assert_eq!(variant.moments,plan.moments);
        assert!(!Arc::ptr_eq(&variant.execution_cache,&plan.execution_cache));
        assert!(!Arc::ptr_eq(&variant.variant_cache,&plan.variant_cache));
        eprintln!("packed-G structural test: analytic variant reused exact parent graph with fresh caches");
        assert_eq!(variant.layout.scalar_products,Some(24..45));assert_eq!(variant.thin_six_child_rank,Some(0));
        assert_eq!(variant.analytic_jet_order,2);assert_eq!(variant.external_primal_seeds,plan.external_primal_seeds);
        assert_eq!(serde_json::to_value(variant.settings()).unwrap(),serde_json::to_value(settings).unwrap());
        assert!(plan.variant(PlanVariant {analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,
            certified_planar_mask:0,certified_affine_chart:None,thin_six_child_rank:None,thin_seven_child_rank:None,
            certified_invariant_chart:None}).is_err());
    }

    #[test]
    #[ignore = "real N6 supplied-G tensor symbolic preparation; requires a separate finite source-reviewed run"]
    fn checked_tensor_proof_mode_survives_full_dual_variant_rebuild() {
        let mut settings = ReductionSettings::new(6,1);
        settings.powers[0] = 2;
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        settings.algorithm = Algorithm::PassarinoVeltman;
        let plan = ReductionPlan::new_certified_tensor(settings,[1,2,3,4]).unwrap();
        let variant = plan.with_analytic_jet_order(2).unwrap();
        let again = plan.with_analytic_jet_order(2).unwrap();
        assert!(Arc::ptr_eq(&variant,&again));
        assert_eq!(variant.certified_invariant_chart,Some(CertifiedInvariantChart::CartesianTensor([1,2,3,4])));
        assert_eq!(variant.settings.rank,1);
        assert_eq!(variant.settings.powers,vec![2,1,1,1,1,1]);
        assert_eq!(variant.algorithm_used(),Algorithm::Ibp);
        assert_eq!(variant.layout.scalar_products,Some(24..45));
        assert_eq!(variant.parameters,plan.parameters);
        assert_eq!(variant.external_primal_seeds,plan.external_primal_seeds);
        let mut wrong = PlanVariant {
            analytic_jet_order:2,bubble_equal_masses:None,zero_invariants:None,
            certified_planar_mask:0, certified_affine_chart: None,thin_six_child_rank:None,thin_seven_child_rank:None,
            certified_invariant_chart:Some(CertifiedInvariantChart::Scalar([1,2,3,4])),
        };
        assert!(matches!(plan.variant(wrong.clone()),Err(ReductionError::Unsupported(_))));
        wrong.certified_invariant_chart=None;
        assert!(matches!(plan.variant(wrong),Err(ReductionError::Unsupported(_))));
    }
}
