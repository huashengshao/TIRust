//! Exact topology preparation. No sampled values are cached in a plan.

use crate::{
    basis::{Monomial, tensor_basis},
    convention::convert_convention,
    ibp::{MasterLeaf, ScalarCompiler, ScalarRequest},
    pv::PvCompiler,
    types::{Algorithm, InputLayout, InvariantInput, ReductionError, ReductionSettings},
};
use std::collections::HashMap;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, integer::Integer},
};

#[derive(Clone, Debug)]
pub struct ReductionPlan {
    pub(crate) settings: ReductionSettings,
    pub(crate) basis: Vec<Monomial>,
    pub(crate) parameters: Vec<Atom>,
    pub(crate) layout: InputLayout,
    pub(crate) distances: Vec<Vec<Atom>>,
    pub(crate) moments: Vec<[Atom; 4]>,
    pub(crate) leaves: Vec<MasterLeaf>,
    pub(crate) master_arguments: Vec<Vec<Atom>>,
    pub(crate) aliases: Vec<(Atom, Atom)>,
    pub(crate) fallback: Option<Box<ReductionPlan>>,
    algorithm_used: Algorithm,
    preparation_notes: Vec<String>,
    pub(crate) special_kinematics: bool,
    pub(crate) analytic_jet_order: u16,
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<ZeroInvariantProfile>,
}

/// Exact constraints are structural: numerical values are never stored here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ZeroInvariantProfile {
    pub masses: Vec<bool>,
    pub distances: Vec<(usize, usize)>,
}

impl ReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0)
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
        Self::build(settings, Some((jet_order, equal_masses)), None, jet_order)
    }

    pub(crate) fn with_analytic_jet_order(&self, order: u16) -> Result<Self, ReductionError> {
        let bubble_limit = self.bubble_limit.map(|(_, equal)| (order, equal));
        Self::build(
            self.settings.clone(),
            bubble_limit,
            self.zero_invariants.clone(),
            order,
        )
    }

    /// Prepare an exact invariant profile. The runtime must verify each selected
    /// constraint is zero in every Taylor component before dispatching a lane.
    pub(crate) fn with_zero_invariants(
        &self,
        mut profile: ZeroInvariantProfile,
    ) -> Result<Self, ReductionError> {
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
        Self::build(
            self.settings.clone(),
            None,
            Some(profile),
            self.analytic_jet_order,
        )
    }

    fn build(
        settings: ReductionSettings,
        bubble_limit: Option<(u16, bool)>,
        zero_invariants: Option<ZeroInvariantProfile>,
        analytic_jet_order: u16,
    ) -> Result<Self, ReductionError> {
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
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..n).map(|i| variable(&format!("m2_{i}"))).collect();
        let mu = variable("mu");
        let mut parameters: Vec<Atom> = shifts.iter().flatten().cloned().collect();
        let mut distances = vec![vec![Atom::new(); n]; n];
        let scalar_products = match settings.invariants {
            InvariantInput::FromShifts => {
                for i in 0..n {
                    for j in i + 1..n {
                        distances[i][j] = squared_difference(&shifts[i], &shifts[j]);
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
        if !ordinary && settings.algorithm == Algorithm::PassarinoVeltman {
            return Err(ReductionError::Unsupported("PV currently requires unit propagator powers and zero dimension shift; use IBP or hybrid".into()));
        }
        let mut preparation_notes = Vec::new();
        let (mut moments, leaves, aliases, algorithm_used) = match algorithm_used {
            Algorithm::Ibp => {
                match compile_ibp(
                    &settings,
                    &basis,
                    &shifts,
                    &masses,
                    &distances,
                    bubble_limit,
                    zero_invariants.as_ref(),
                ) {
                    Ok((moments, leaves, aliases)) => (moments, leaves, aliases, Algorithm::Ibp),
                    Err(error) if ordinary && n >= 5 && !special_kinematics => {
                        // Original IBP also completes its four-dimensional
                        // high-point tensor system with PV at zero dimension
                        // coefficients. This does not replace an independent
                        // higher-dimensional scalar integral by box masters.
                        preparation_notes.push(format!("IBP tensor completion uses PV: {error}"));
                        let (moments, leaves, aliases) =
                            compile_pv(&basis, &shifts, &masses, &distances, analytic_jet_order)?;
                        (moments, leaves, aliases, Algorithm::PassarinoVeltman)
                    }
                    Err(error) => return Err(error),
                }
            }
            Algorithm::PassarinoVeltman | Algorithm::Hybrid => {
                let pv = compile_pv(&basis, &shifts, &masses, &distances, analytic_jet_order);
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
                        )?;
                        (moments, leaves, aliases, Algorithm::Ibp)
                    }
                    Err(error) => return Err(error),
                }
            }
        };
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
                match active.len() {
                    1 => {}
                    2 => distance(0, 1),
                    3 => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(0, 2);
                    }
                    4 => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(2, 3);
                        distance(0, 3);
                        distance(0, 2);
                        distance(1, 3);
                    }
                    _ => unreachable!("scalar compiler only emits one through four point masters"),
                }
                args.extend(active.iter().map(|&i| masses[i].clone()));
                args.push(mu.pow(2));
                args
            })
            .collect();
        let fallback = if !special_kinematics
            && settings.algorithm == Algorithm::Hybrid
            && algorithm_used == Algorithm::PassarinoVeltman
        {
            let mut fallback_settings = settings.clone();
            fallback_settings.algorithm = Algorithm::Ibp;
            match Self::build(fallback_settings, None, None, analytic_jet_order) {
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
            settings,
            basis,
            parameters,
            layout,
            distances,
            moments,
            leaves,
            master_arguments,
            aliases,
            fallback,
            algorithm_used,
            preparation_notes,
            special_kinematics,
            analytic_jet_order,
            bubble_limit,
            zero_invariants,
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
}

fn compile_pv(
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    distances: &[Vec<Atom>],
    analytic_jet_order: u16,
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
    let mut compiler =
        PvCompiler::with_distances(shifts.to_vec(), masses.to_vec(), distances.to_vec(), true)
            .map_err(ReductionError::Preparation)?;
    compiler
        .set_analytic_jet_order(analytic_jet_order)
        .map_err(ReductionError::Preparation)?;
    let moments = basis
        .iter()
        .map(|m| compiler.compile_moment(m.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ReductionError::Preparation)?;
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
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
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
    compiler
        .enable_aliases()
        .map_err(ReductionError::Preparation)?;
    compiler
        .enable_physical_poles()
        .map_err(ReductionError::Preparation)?;
    let t: [Atom; 4] = std::array::from_fn(|i| variable(&format!("aux{i}")));
    let metric = t[0].pow(2) - t[1].pow(2) - t[2].pow(2) - t[3].pow(2);
    let projections: Vec<Atom> = shifts
        .iter()
        .map(|p| t.iter().zip(p).map(|(a, b)| a * b).sum())
        .collect();
    let mut by_exponent: HashMap<[u16; 4], [Atom; 4]> = HashMap::new();
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
                let value = compiler
                    .compile(ScalarRequest {
                        powers,
                        dimension_shift: scalar_h,
                    })
                    .map_err(ReductionError::Preparation)?;
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
    Ok((moments, compiler.leaves, aliases))
}

fn multinomial(exponents: [u16; 4]) -> Atom {
    let rank: u32 = exponents.iter().map(|&e| u32::from(e)).sum();
    exponents
        .into_iter()
        .fold(Atom::num(Integer::factorial(rank)), |value, e| {
            value / Atom::num(Integer::factorial(u32::from(e)))
        })
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi::{name}")))
}
fn squared_difference(a: &[Atom; 4], b: &[Atom; 4]) -> Atom {
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
