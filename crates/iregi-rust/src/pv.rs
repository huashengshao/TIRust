//! Passarino--Veltman tensor recurrences compiled with Symbolica.
//!
//! For up to four denominators the coefficients obey Denner--Dittmaier
//! Eqs. (5.10)--(5.11), including the epsilon-dependent trace denominator.
//! Five-point Cartesian moments use the complete four-dimensional external
//! basis or an exact constant denominator identity on a constrained family;
//! six/seven-point moments use exact denominator identities.
//! This avoids introducing a dimension-shifted pentagon divided by epsilon.
//! Symbolica owns matrix solution, expression arithmetic and polynomial
//! collection. Integer compositions below enumerate tensor indices only.

use std::{collections::HashMap, time::Instant};

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::{atom::AtomField, integer::Integer},
    id::Replacement,
    tensors::matrix::Matrix,
};

mod thin_seven;
pub(crate) use thin_seven::{ThinSevenChart, ThinSevenGeometry, ThinSixGeometry};

use crate::ibp::{MasterLeaf, ScalarCompiler, ScalarRequest};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CoefficientKey {
    active: Vec<usize>,
    metrics: u16,
    momenta: Vec<u16>,
}

#[derive(Clone)]
struct Geometry {
    shifts: Vec<[Atom; 4]>,
    f: Vec<Atom>,
    inverse_z: Vec<Vec<Atom>>,
}

#[derive(Clone)]
struct FivePointRelation {
    active: Atom,
    coefficients: Vec<Atom>,
}

/// A rank constraint required by a lazily selected coordinate identity.
/// The executor must certify the raw polynomial on exact represented inputs;
/// a zero obtained through floating-point cancellation is insufficient.
#[derive(Clone, Debug)]
pub(crate) struct CoordinateConstraint {
    /// Alias whose graph reachability determines whether this topology is used.
    pub owner: Atom,
    /// Nonzero only when the rank-deficient identity branch is selected.
    pub active: Atom,
    /// Raw-coordinate polynomial, without aliases or master dependence.
    pub expression: Atom,
    pub label: String,
}

/// An independent PV compiler for unit propagator powers in D=4-2epsilon.
/// The returned tensors have four-dimensional Cartesian numerator indices.
/// Values use the same native four Laurent slots as [`ScalarCompiler`].
pub struct PvCompiler {
    shifts: Vec<[Atom; 4]>,
    mass_squared: Vec<Atom>,
    distances: Vec<Vec<Atom>>,
    substitutions: Vec<Replacement>,
    numeric_kinematics: bool,
    alias_enabled: bool,
    profile_preparation: bool,
    aliases: Vec<(Atom, Atom)>,
    scalar: ScalarCompiler,
    bubble_expansions: HashMap<(Vec<usize>, bool), ScalarCompiler>,
    analytic_jet_order: u16,
    coordinate_identities: bool,
    /// Private execution chart; its full Taylor constraints are checked by the wrapper.
    certified_planar_chart: Option<usize>,
    certified_planar_mask: u8,
    preparation_notes: Vec<String>,
    coordinate_constraints: Vec<CoordinateConstraint>,
    gram_scalar_product_transform: Option<Box<dyn FnMut(Atom) -> Atom>>,
    coefficients: HashMap<CoefficientKey, [Atom; 4]>,
    geometry: HashMap<Vec<usize>, Geometry>,
    moments: HashMap<(Vec<usize>, [u16; 4]), [Atom; 4]>,
    generated: HashMap<(Vec<usize>, u16), HashMap<Atom, [Atom; 4]>>,
    denominator_identities: HashMap<Vec<usize>, Vec<Atom>>,
    five_point_relations: HashMap<Vec<usize>, Option<FivePointRelation>>,
    z: [Atom; 4],
}

impl PvCompiler {
    /// Inline compiler, useful for independent inspection of exact expressions.
    pub fn new(shifts: Vec<[Atom; 4]>, mass_squared: Vec<Atom>) -> Result<Self, String> {
        Self::from_coordinates(shifts, mass_squared, false)
    }

    /// Compile a dependency-ordered Symbolica DAG rather than inline trees.
    /// Pass `aliases()` to Symbolica's evaluator builder `add_aliases` and use
    /// `direct_translation(true)` when lowering the returned moment handles.
    pub fn with_aliases(shifts: Vec<[Atom; 4]>, mass_squared: Vec<Atom>) -> Result<Self, String> {
        Self::from_coordinates(shifts, mass_squared, true)
    }

    fn from_coordinates(
        shifts: Vec<[Atom; 4]>,
        mass_squared: Vec<Atom>,
        alias_enabled: bool,
    ) -> Result<Self, String> {
        let distances = shifts
            .iter()
            .map(|p| {
                shifts
                    .iter()
                    .map(|q| {
                        let delta = std::array::from_fn(|mu| &p[mu] - &q[mu]);
                        dot(&delta, &delta)
                    })
                    .collect()
            })
            .collect();
        let mut compiler = Self::with_distances(shifts, mass_squared, distances, alias_enabled)?;
        compiler.coordinate_identities = true;
        Ok(compiler)
    }

    /// Use prescribed squared distances, as required by IREGI's supplied
    /// scalar-product interface. Coordinates only assemble Cartesian tensors;
    /// every Gram/Cayley matrix and scalar master uses `distances`.
    pub fn with_distances(
        shifts: Vec<[Atom; 4]>,
        mass_squared: Vec<Atom>,
        mut distances: Vec<Vec<Atom>>,
        alias_enabled: bool,
    ) -> Result<Self, String> {
        let n = shifts.len();
        if n == 0
            || n != mass_squared.len()
            || distances.len() != n
            || distances.iter().any(|row| row.len() != n)
        {
            return Err(
                "PV needs a nonempty shift/mass list and matching square distance matrix".into(),
            );
        }
        for i in 0..n {
            if !distances[i][i].together().is_zero() {
                return Err("PV squared-distance matrix must have zero diagonal".into());
            }
            for j in i + 1..n {
                if !(&distances[i][j] - &distances[j][i]).together().is_zero() {
                    return Err("PV squared-distance matrix must be symmetric".into());
                }
            }
        }
        let symbolic = distances
            .iter()
            .flatten()
            .any(|x| !matches!(x.as_view(), AtomView::Num(_)));
        let numeric_kinematics = !alias_enabled
            && !symbolic
            && shifts
                .iter()
                .flatten()
                .chain(&mass_squared)
                .all(|x| matches!(x.as_view(), AtomView::Num(_)));
        let mut substitutions = Vec::new();
        let mut aliases = Vec::new();
        if symbolic {
            for i in 0..n {
                for j in i + 1..n {
                    if matches!(distances[i][j].as_view(), AtomView::Num(_)) {
                        continue;
                    }
                    let alias = Atom::var(symbolica::symbol!(format!("iregi::pv_s{i}_{j}")));
                    let distance = distances[i][j].clone();
                    substitutions.push(Replacement::new(alias.clone(), distance.clone()));
                    aliases.push((alias.clone(), distance));
                    distances[i][j] = alias.clone();
                    distances[j][i] = alias;
                }
            }
        }
        let r = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| &distances[i][j] - &mass_squared[i] - &mass_squared[j])
                    .collect()
            })
            .collect();
        let mut scalar = ScalarCompiler::new(r)?;
        if alias_enabled {
            scalar.enable_physical_poles()?;
        }
        Ok(Self {
            shifts,
            mass_squared,
            distances,
            substitutions,
            numeric_kinematics,
            alias_enabled,
            profile_preparation: std::env::var_os("IREGI_PV_PREPARATION_PROFILE").is_some(),
            aliases,
            scalar,
            bubble_expansions: HashMap::new(),
            analytic_jet_order: 0,
            coordinate_identities: false,
            certified_planar_chart: None,
            certified_planar_mask: 0,
            preparation_notes: Vec::new(),
            coordinate_constraints: Vec::new(),
            gram_scalar_product_transform: None,
            coefficients: HashMap::new(),
            geometry: HashMap::new(),
            moments: HashMap::new(),
            generated: HashMap::new(),
            denominator_identities: HashMap::new(),
            five_point_relations: HashMap::new(),
            z: std::array::from_fn(|mu| Atom::var(symbolica::symbol!(format!("iregi::pv_z{mu}")))),
        })
    }

    /// Set the total nilpotent degree retained at analytic lightlike bubble
    /// branches. Call before requesting any moments. Ordinary numeric types
    /// use zero; a Dual plan must use its full algebra's maximal total degree.
    pub fn set_analytic_jet_order(&mut self, order: u16) -> Result<(), String> {
        if !self.coefficients.is_empty() || !self.moments.is_empty() {
            return Err("set the PV analytic jet order before compiling moments".into());
        }
        self.analytic_jet_order = order;
        Ok(())
    }

    /// Permit identities derived from the supplied four-vector coordinates.
    /// Enable only when invariants are derived from those same coordinates;
    /// arbitrary authoritative scalar products do not satisfy this contract.
    pub fn set_coordinate_identities(&mut self, enabled: bool) -> Result<(), String> {
        if !self.moments.is_empty() || !self.coefficients.is_empty() {
            return Err("set coordinate identity policy before compiling moments".into());
        }
        self.coordinate_identities = enabled;
        Ok(())
    }

    /// Select a constant-one pentagon chart only behind exact input certification.
    /// No plan containing this policy may escape the checked wrapper.
    pub(crate) fn set_certified_planar_chart(&mut self, omitted: usize) -> Result<(), String> {
        if self.shifts.len() != 5
            || omitted >= 4
            || !self.moments.is_empty()
            || !self.coefficients.is_empty()
        {
            return Err("certified planar chart needs five propagators, an omitted coordinate row in 0..4, and an unused compiler".into());
        }
        self.certified_planar_chart = Some(omitted);
        self.coordinate_identities = true;
        Ok(())
    }

    /// Private wrapper-only policy. A mask has no meaning without the full
    /// planar and per-cofactor Taylor certificates checked before execution.
    pub(crate) fn set_certified_planar_mask(&mut self, mask:u8)->Result<(),String> {
        if mask >= 31 || self.certified_planar_chart.is_none()
            || self.shifts.len()!=5 || !self.moments.is_empty() || !self.coefficients.is_empty() {
            return Err("certified planar mask requires an unused five-point checked chart and at least one active child".into());
        }
        self.certified_planar_mask=mask;
        Ok(())
    }

    /// Exact-domain limitations encountered while constructing this plan.
    pub fn preparation_notes(&self) -> &[String] {
        &self.preparation_notes
    }

    pub(crate) fn coordinate_constraints(&self) -> &[CoordinateConstraint] {
        &self.coordinate_constraints
    }

    /// Override each raw `(p_i-p_origin) dot (p_j-p_origin)` used by PV
    /// Gram matrices. Intended for explicit legacy on-shell snapping only;
    /// the default and supplied-scalar-product interfaces preserve inputs.
    /// Any aliases emitted by the transform must be registered before the
    /// PV aliases, since a recurrence can depend on a transformed product.
    pub fn set_gram_scalar_product_transform(
        &mut self,
        transform: impl FnMut(Atom) -> Atom + 'static,
    ) -> Result<(), String> {
        if !self.coefficients.is_empty() || !self.moments.is_empty() || !self.geometry.is_empty() {
            return Err("set the PV Gram transform before compiling moments".into());
        }
        self.gram_scalar_product_transform = Some(Box::new(transform));
        Ok(())
    }

    /// Shared exact definitions in dependency order; empty for an inline compiler.
    pub fn aliases(&self) -> &[(Atom, Atom)] {
        if self.alias_enabled {
            &self.aliases
        } else {
            &[]
        }
    }

    fn cache_value(&mut self, values: [Atom; 4]) -> [Atom; 4] {
        if self.alias_enabled {
            values.map(|body| self.bind_alias(body))
        } else if self.numeric_kinematics {
            // Exact numeric coefficients collapse to a short master-linear sum.
            let started = self.profile_preparation.then(Instant::now);
            let values = values.map(|value| value.expand());
            if let Some(started) = started {
                let elapsed = started.elapsed();
                if elapsed.as_millis() >= 10 {
                    eprintln!(
                        "PV prepare exact master-linear expansion: {elapsed:?}, {} moments and {} coefficients cached",
                        self.moments.len(),
                        self.coefficients.len()
                    );
                }
            }
            values
        } else {
            values
        }
    }

    fn bind_alias(&mut self, body: Atom) -> Atom {
        if !self.alias_enabled || matches!(body.as_view(), AtomView::Num(_) | AtomView::Var(_)) {
            return body;
        }
        let alias = Atom::var(symbolica::symbol!(format!(
            "iregi::pv_node_{}",
            self.aliases.len()
        )));
        self.aliases.push((alias.clone(), body));
        alias
    }

    pub fn leaves(&self) -> &[MasterLeaf] {
        &self.scalar.leaves
    }

    pub fn node_count(&self) -> usize {
        self.coefficients.len()
            + self.moments.len()
            + self.scalar.node_count()
            + self
                .bubble_expansions
                .values()
                .map(ScalarCompiler::node_count)
                .sum::<usize>()
    }

    /// Private six-point thin dispatcher. Column zero is the constant-one
    /// identity; columns one through four lower a Cartesian numerator index.
    /// One Symbolica inverse solves all five right-hand sides. No tensor or
    /// scalar child recurrence is generated here.
    pub(crate) fn thin_six_weights(&mut self) -> Result<[Vec<Atom>; 5], String> {
        if self.shifts.len() != 6 || !self.moments.is_empty() {
            return Err("thin six-point weights require an unused six-point compiler".into());
        }
        let active: Vec<_> = (0..6).collect();
        let mut rows = coordinate_rows(&self.shifts, &active);
        rows.push(self.f_values(&active));
        let matrix = Matrix::from_nested_vec(rows, field())
            .map_err(|error| format!("thin six-point matrix: {error}"))?;
        let determinant = matrix.det().map_err(|error| format!("thin six-point determinant: {error:?}"))?;
        if determinant.is_zero() {
            return Err("thin six-point coordinate chart is structurally singular; same-rank fallback is unsupported".into());
        }
        let rhs = Matrix::from_nested_vec((0..5).map(|row| (0..5).map(|column| {
            if column == 0 { Atom::num(i64::from(row == 4)) }
            else if row == 4 { -&self.shifts[0][column-1] }
            else if row == column-1 { Atom::num(if row == 0 { 1 } else { -1 }) / Atom::num(2) }
            else { Atom::new() }
        }).collect()).collect(), field()).map_err(|error| format!("thin six-point RHS: {error}"))?;
        let inverse = matrix.inv().map_err(|error| format!("thin six-point inverse: {error:?}"))?;
        let solutions = &inverse * &rhs;
        Ok(std::array::from_fn(|column| {
            let mut weights = vec![Atom::new(); 6];
            for row in 0..5 { weights[row+1] = solutions[(row as u32, column as u32)].clone(); }
            weights[0] = -Atom::add_many(&weights[1..]);
            weights.into_iter().map(|weight| self.bind_alias(weight)).collect()
        }))
    }

    /// Compile one raw Cartesian monomial; there is no multinomial weight.
    pub fn compile_moment(&mut self, exponents: [u16; 4]) -> Result<[Atom; 4], String> {
        let active: Vec<_> = (0..self.shifts.len()).collect();
        let value = self.moment(&active, exponents)?;
        if self.alias_enabled {
            Ok(value)
        } else {
            Ok(value.map(|expression| expression.replace_multiple(self.substitutions.iter())))
        }
    }

    /// Convert an eligible generalized scalar to a canonical PV coefficient.
    ///
    /// With e=sum(nu_i-1) and k=h-e, IREGI's IBP2PAVE relation is
    /// J_h(nu)=(-2)^k C_{00...0,1...}/prod((nu_i-1)!). The internal
    /// normalization has already removed pi^h. Eligibility is tested for each
    /// generated scalar request, including those in negative-shift tensors.
    /// `None` means the original generalized interface selects IBP instead.
    ///
    /// Canonical metric coefficients are independent only through four
    /// propagators here. In particular, a Cartesian pentagon moment cannot
    /// substitute for the independent genuine D6 scalar master.
    pub(crate) fn compile_generalized_scalar(
        &mut self,
        request: &ScalarRequest,
    ) -> Result<Option<[Atom; 4]>, String> {
        if request.powers.len() != self.shifts.len()
            || request.powers.iter().any(|&power| power == 0)
        {
            return Err("generalized PV requires one positive power per propagator".into());
        }
        let excess: i32 = request
            .powers
            .iter()
            .map(|&power| i32::from(power) - 1)
            .sum();
        let metrics = i32::from(request.dimension_shift) - excess;
        if request.powers[0] != 1 || metrics < 0 {
            return Ok(None);
        }
        if self.shifts.len() > 4 {
            return Err("generalized canonical PV coefficients beyond four propagators require an independent shifted-master completion".into());
        }
        let mapped_rank = 2 * metrics + excess;
        if mapped_rank > 32 {
            return Err(
                "generalized PV coefficient exceeds the 32-index preparation budget".into(),
            );
        }
        let metrics = u16::try_from(metrics)
            .map_err(|_| "generalized PV metric count is outside the supported integer range")?;
        let momenta: Vec<_> = request.powers[1..].iter().map(|&power| power - 1).collect();
        let active: Vec<_> = (0..self.shifts.len()).collect();
        let coefficient = self.coefficient(&active, metrics, &momenta)?;
        let factor = Atom::num(-2).pow(i64::from(metrics))
            / request
                .powers
                .iter()
                .map(|&power| factorial(power - 1))
                .product::<Atom>();
        let value = scale(&coefficient, &factor);
        Ok(Some(if self.alias_enabled {
            value
        } else {
            value.map(|expression| expression.replace_multiple(self.substitutions.iter()))
        }))
    }

    fn scalar(&mut self, active: &[usize]) -> Result<[Atom; 4], String> {
        let mut powers = vec![0; self.shifts.len()];
        for &i in active {
            powers[i] = 1;
        }
        self.scalar.compile(ScalarRequest {
            powers,
            dimension_shift: 0,
        })
    }

    fn moment(&mut self, active: &[usize], exponents: [u16; 4]) -> Result<[Atom; 4], String> {
        let key = (active.to_vec(), exponents);
        if let Some(value) = self.moments.get(&key) {
            return Ok(value.clone());
        }
        let rank: u16 = exponents.iter().sum();
        let result = if active.is_empty() {
            zero()
        } else if active.len() >= 6 {
            // This exact identity also applies at rank zero and does not rely
            // on inverting the identically singular N>=7 Cayley matrix.
            let factors = self.denominator_identity(active)?;
            let mut result = zero();
            for (i, factor) in factors.iter().enumerate() {
                if factor.is_zero() {
                    continue;
                }
                let child = without(active, i);
                let value = self.moment(&child, exponents)?;
                accumulate(&mut result, factor, &value);
            }
            result
        } else if active.len() == 5 {
            self.five_point_moment(active, exponents)?
        } else if rank == 0 {
            self.scalar(active)?
        } else {
            self.generate_rank(active, rank)?;
            let monomial = monomial(&self.z, &exponents);
            let weighted = self.generated[&(active.to_vec(), rank)]
                .get(&monomial)
                .cloned()
                .unwrap_or_else(zero);
            let divisor =
                factorial(rank) / exponents.iter().map(|&n| factorial(n)).product::<Atom>();
            scale(&weighted, &(Atom::num(1) / divisor))
        };
        let result = self.cache_value(result);
        self.moments.insert(key, result.clone());
        Ok(result)
    }

    fn five_point_moment(
        &mut self,
        active: &[usize],
        exponents: [u16; 4],
    ) -> Result<[Atom; 4], String> {
        let relation = if self.coordinate_identities {
            self.five_point_constant_relation(active)?
        } else {
            None
        };
        let pinched = if let Some(relation) = &relation {
            let mut value = zero();
            for (i, coefficient) in relation.coefficients.iter().enumerate() {
                if !coefficient.is_zero() {
                    let child = self.moment(&without(active, i), exponents)?;
                    accumulate(&mut value, coefficient, &child);
                }
            }
            if relation.active == Atom::num(1) {
                return Ok(value);
            }
            Some(value)
        } else {
            None
        };
        let ordinary = if exponents.iter().all(|&n| n == 0) {
            self.scalar(active)?
        } else {
            let g = self.geometry(active)?;
            let mu = exponents.iter().position(|&n| n > 0).unwrap();
            let mut rest = exponents;
            rest[mu] -= 1;
            let base = self.moment(active, rest)?;
            let pinch_zero = self.moment(&without(active, 0), rest)?;
            let mut result = scale(&base, &(-&self.shifts[active[0]][mu]));
            for j in 0..4 {
                let factor = (0..4).fold(Atom::new(), |sum, i| {
                    sum + &g.shifts[i][mu] * &g.inverse_z[i][j]
                });
                let pinch_j = self.moment(&without(active, j + 1), rest)?;
                let mut rhs = pinch_j;
                accumulate(&mut rhs, &Atom::num(-1), &pinch_zero);
                accumulate(&mut rhs, &(-&g.f[j]), &base);
                accumulate(&mut result, &factor, &rhs);
            }
            result
        };
        Ok(
            if let (Some(relation), Some(pinched)) = (relation, pinched) {
                std::array::from_fn(|slot| {
                    conditional(&relation.active, &pinched[slot], &ordinary[slot])
                })
            } else {
                ordinary
            },
        )
    }

    /// Generate the polynomial integral of `(z_mu q^mu)^rank` and let
    /// Symbolica collect its four Cartesian indeterminates.
    fn generate_rank(&mut self, active: &[usize], rank: u16) -> Result<(), String> {
        let key = (active.to_vec(), rank);
        if self.generated.contains_key(&key) {
            return Ok(());
        }
        let _generation = PreparationTimer::new(
            self.profile_preparation,
            format_args!("rank polynomial {active:?}, rank {rank}"),
        );
        let n = active.len();
        let origin = self.shifts[active[0]].clone();
        let translated: Vec<[Atom; 4]> = active[1..]
            .iter()
            .map(|&i| std::array::from_fn(|mu| &self.shifts[i][mu] - &origin[mu]))
            .collect();
        let projected: Vec<_> = translated
            .iter()
            .map(|p| cartesian_dot(p, &self.z))
            .collect();
        let translation = -cartesian_dot(&origin, &self.z);
        let metric = dot(&self.z, &self.z);
        let mut expression = zero();
        for local_rank in 0..=rank {
            let translation_factor =
                pow(&translation, rank - local_rank) * binomial(rank, local_rank);
            for metrics in 0..=local_rank / 2 {
                let momentum_rank = local_rank - 2 * metrics;
                for momenta in compositions(momentum_rank, n - 1) {
                    let coefficients = self.coefficient(active, metrics, &momenta)?;
                    let multiplicity = factorial(local_rank)
                        / (Atom::num(2).pow(i64::from(metrics))
                            * factorial(metrics)
                            * momenta.iter().map(|&p| factorial(p)).product::<Atom>());
                    let factor = &translation_factor
                        * pow(&metric, metrics)
                        * monomial(&projected, &momenta)
                        * multiplicity;
                    accumulate(&mut expression, &factor, &coefficients);
                }
            }
        }
        let mut collected: HashMap<Atom, [Atom; 4]> = HashMap::new();
        for slot in 0..4 {
            let _collection = PreparationTimer::new(
                self.profile_preparation,
                format_args!(
                    "polynomial coefficient collection {active:?}, rank {rank}, slot {slot}"
                ),
            );
            for (key, coefficient) in expression[slot].coefficient_list::<u16>(&self.z) {
                collected.entry(key).or_insert_with(zero)[slot] = coefficient;
            }
        }
        self.generated.insert(key, collected);
        Ok(())
    }

    fn coefficient(
        &mut self,
        active: &[usize],
        metrics: u16,
        momenta: &[u16],
    ) -> Result<[Atom; 4], String> {
        debug_assert_eq!(momenta.len() + 1, active.len());
        let key = CoefficientKey {
            active: active.to_vec(),
            metrics,
            momenta: momenta.to_vec(),
        };
        if let Some(value) = self.coefficients.get(&key) {
            return Ok(value.clone());
        }
        let rank: u16 = 2 * metrics + momenta.iter().sum::<u16>();
        // A lightlike bubble has no invertible external Gram matrix. Its
        // tensor coefficients are exact dimension-shifted scalar integrals.
        // This handles pinched bubbles as well as an entire bubble topology.
        let literal_lightlike = active.len() == 2 && self.distances[active[0]][active[1]].is_zero();
        let mut result = if literal_lightlike && rank > 0 {
            self.lightlike_coefficient(active, metrics, momenta[0])?
        } else if rank == 0 {
            self.scalar(active)?
        } else if active.len() == 1 {
            let lower = self.coefficient(active, metrics - 1, &[])?;
            scale(
                &divide_dimension(&lower, 2 * i32::from(metrics) + 2),
                &self.mass_squared[active[0]],
            )
        } else if metrics > 0 {
            // T_00... = [T(pinch 0) + 2 m0² T + sum f_i T_i]
            //            / [2(D + rank - N - 1)].
            let mut result = self.shifted_pinch_zero(active, metrics - 1, momenta)?;
            let lower = self.coefficient(active, metrics - 1, momenta)?;
            accumulate(
                &mut result,
                &(Atom::num(2) * &self.mass_squared[active[0]]),
                &lower,
            );
            let f = self.f_values(active);
            for i in 0..momenta.len() {
                let mut child = momenta.to_vec();
                child[i] += 1;
                let value = self.coefficient(active, metrics - 1, &child)?;
                accumulate(&mut result, &f[i], &value);
            }
            let c = i32::from(rank) + 3 - active.len() as i32;
            if c == 0 {
                return Err("PV trace recurrence needs a nonzero constant divisor".into());
            }
            scale(
                &divide_dimension(&result, c),
                &(Atom::num(1) / Atom::num(2)),
            )
        } else {
            let g = self.geometry(active)?;
            let j = momenta.iter().position(|&n| n != 0).unwrap();
            let mut rest = momenta.to_vec();
            rest[j] -= 1;
            let lower = self.coefficient(active, 0, &rest)?;
            let pinch_zero = self.shifted_pinch_zero(active, 0, &rest)?;
            let mut result = zero();
            for i in 0..momenta.len() {
                let mut rhs = scale(&pinch_zero, &Atom::num(-1));
                accumulate(&mut rhs, &(-&g.f[i]), &lower);
                if rest[i] == 0 {
                    let mut child_indices = rest.clone();
                    child_indices.remove(i);
                    let value = self.coefficient(&without(active, i + 1), 0, &child_indices)?;
                    accumulate(&mut rhs, &Atom::num(1), &value);
                } else {
                    let mut child = rest.clone();
                    child[i] -= 1;
                    let value = self.coefficient(active, 1, &child)?;
                    accumulate(&mut rhs, &Atom::num(-2 * i32::from(rest[i])), &value);
                }
                accumulate(&mut result, &g.inverse_z[j][i], &rhs);
            }
            result
        };
        if active.len() == 2 && rank > 0 && !literal_lightlike {
            let distance = self.distances[active[0]][active[1]].clone();
            if !matches!(distance.as_view(), AtomView::Num(_)) {
                let lightlike = self.lightlike_coefficient(active, metrics, momenta[0])?;
                result = std::array::from_fn(|slot| {
                    conditional(&distance, &result[slot], &lightlike[slot])
                });
            }
        }
        // With constant kinematics, Symbolica can collect the linear master
        // coefficients exactly and cheaply. This prevents nested recurrence
        // trees from growing exponentially during specialized preparation.
        let result = self.cache_value(result);
        self.coefficients.insert(key, result.clone());
        Ok(result)
    }

    fn lightlike_coefficient(
        &mut self,
        active: &[usize],
        metrics: u16,
        momentum_count: u16,
    ) -> Result<[Atom; 4], String> {
        let mass_difference =
            (&self.mass_squared[active[1]] - &self.mass_squared[active[0]]).together();
        let mass = self.mass_squared[active[0]].clone();
        let exact = matches!(mass_difference.as_view(), AtomView::Num(_));
        let unequal = if !mass_difference.is_zero() {
            Some(self.lightlike_expansion(active, metrics, momentum_count, false)?)
        } else {
            None
        };
        if exact && !mass_difference.is_zero() {
            return Ok(unequal.unwrap());
        }
        let scaleless = if metrics == 0 {
            let residue = Atom::num(if momentum_count % 2 == 0 { 1 } else { -1 })
                / Atom::num(i64::from(momentum_count) + 1);
            [residue.clone(), -residue, Atom::new(), Atom::new()]
        } else {
            zero()
        };
        let equal = if mass.is_zero() {
            scaleless
        } else {
            let value = self.lightlike_expansion(active, metrics, momentum_count, true)?;
            if matches!(mass.as_view(), AtomView::Num(_)) {
                value
            } else {
                std::array::from_fn(|slot| conditional(&mass, &value[slot], &scaleless[slot]))
            }
        };
        Ok(match unequal {
            Some(unequal) => std::array::from_fn(|slot| {
                conditional(&mass_difference, &unequal[slot], &equal[slot])
            }),
            None => equal,
        })
    }

    fn lightlike_expansion(
        &mut self,
        active: &[usize],
        metrics: u16,
        momentum_count: u16,
        equal_masses: bool,
    ) -> Result<[Atom; 4], String> {
        let key = (active.to_vec(), equal_masses);
        let mut compiler = if let Some(compiler) = self.bubble_expansions.remove(&key) {
            compiler
        } else {
            let r = active
                .iter()
                .map(|&i| {
                    active
                        .iter()
                        .map(|&j| {
                            &self.distances[i][j] - &self.mass_squared[i] - &self.mass_squared[j]
                        })
                        .collect()
                })
                .collect();
            let mut compiler =
                ScalarCompiler::lightlike_bubble(r, self.analytic_jet_order, equal_masses)?;
            if self.alias_enabled {
                compiler.enable_physical_poles()?;
            }
            compiler
        };
        let result = compiler.compile(ScalarRequest {
            powers: vec![1, 1 + momentum_count],
            dimension_shift: (metrics + momentum_count) as i16,
        })?;
        // Local scalar compilers use local propagator indices. Register each
        // A0 with the shared global compiler and substitute all four native
        // Laurent slots at once, before caching the PV recurrence node.
        let mut replacements = Vec::new();
        for leaf in &compiler.leaves {
            let global: Vec<_> = leaf.active.iter().map(|&i| active[i]).collect();
            let values = self.scalar(&global)?;
            for (slot, value) in leaf.slots.iter().zip(values) {
                replacements.push(Replacement::new(slot.clone(), value));
            }
        }
        self.bubble_expansions.insert(key, compiler);
        let factor = factorial(momentum_count) * Atom::num(-2).pow(-i64::from(metrics));
        Ok(result.map(|value| value.replace_multiple(replacements.iter()) * &factor))
    }

    /// The coefficient shift in original IREGI's `shiftpaveden2` (Denner 2.9).
    fn shifted_pinch_zero(
        &mut self,
        active: &[usize],
        metrics: u16,
        momenta: &[u16],
    ) -> Result<[Atom; 4], String> {
        let child_active = &active[1..];
        let first = momenta[0];
        let mut result = zero();
        for total in 0..=first {
            for extra in compositions(total, momenta.len() - 1) {
                let child: Vec<_> = momenta[1..]
                    .iter()
                    .zip(&extra)
                    .map(|(&a, &b)| a + b)
                    .collect();
                let value = self.coefficient(child_active, metrics, &child)?;
                let count = binomial(first, total) * factorial(total)
                    / extra.iter().map(|&n| factorial(n)).product::<Atom>();
                let sign = if first % 2 == 0 { 1 } else { -1 };
                accumulate(&mut result, &(Atom::num(sign) * count), &value);
            }
        }
        Ok(result)
    }

    fn f_values(&self, active: &[usize]) -> Vec<Atom> {
        active[1..]
            .iter()
            .map(|&i| {
                &self.distances[active[0]][i] - &self.mass_squared[i]
                    + &self.mass_squared[active[0]]
            })
            .collect()
    }

    fn geometry(&mut self, active: &[usize]) -> Result<Geometry, String> {
        if let Some(geometry) = self.geometry.get(active) {
            return Ok(geometry.clone());
        }
        let origin = &self.shifts[active[0]];
        let shifts: Vec<[Atom; 4]> = active[1..]
            .iter()
            .map(|&i| std::array::from_fn(|mu| &self.shifts[i][mu] - &origin[mu]))
            .collect();
        let rows = if let Some(transform) = &mut self.gram_scalar_product_transform {
            // Original IREGI snaps dot products at this call site, before
            // multiplying by two; reconstructing this matrix from separately
            // snapped squared distances would give different semantics.
            shifts
                .iter()
                .map(|p| {
                    shifts
                        .iter()
                        .map(|q| Atom::num(2) * transform(dot(p, q)))
                        .collect()
                })
                .collect()
        } else {
            active[1..]
                .iter()
                .map(|&i| {
                    active[1..]
                        .iter()
                        .map(|&j| {
                            &self.distances[active[0]][i] + &self.distances[active[0]][j]
                                - &self.distances[i][j]
                        })
                        .collect()
                })
                .collect()
        };
        let matrix =
            Matrix::from_nested_vec(rows, field()).map_err(|e| format!("PV Gram matrix: {e}"))?;
        let _inverse = PreparationTimer::new(
            self.profile_preparation,
            format_args!("Gram inversion {active:?}"),
        );
        let inverse = matrix
            .inv()
            .map_err(|e| format!("PV singular Gram matrix for {active:?}: {e:?}"))?;
        let inverse_z = (0..shifts.len())
            .map(|i| {
                (0..shifts.len())
                    .map(|j| inverse[(i as u32, j as u32)].clone())
                    .collect()
            })
            .collect();
        let geometry = Geometry {
            shifts,
            f: self.f_values(active),
            inverse_z,
        };
        self.geometry.insert(active.to_vec(), geometry.clone());
        Ok(geometry)
    }

    /// Rank-deficient pentagons can admit a constant denominator identity
    /// even though their four-dimensional external Gram matrix is singular.
    /// For generic coordinates, retain four lazy rank-three charts. A chart
    /// is used only at exactly zero coordinate determinant: no small-number
    /// projection or replacement by an approximate relation is made here.
    fn five_point_constant_relation(
        &mut self,
        active: &[usize],
    ) -> Result<Option<FivePointRelation>, String> {
        if let Some(relation) = self.five_point_relations.get(active) {
            return Ok(relation.clone());
        }
        let _relation = PreparationTimer::new(
            self.profile_preparation,
            format_args!("pentagon constant identity {active:?}"),
        );
        let mut rows = coordinate_rows(&self.shifts, active);
        if let Some(omitted) = self.certified_planar_chart {
            if active != [0,1,2,3,4] { return Err("certified planar mask cannot be transferred to a different active subset".into()); }
            rows.push(self.f_values(active));
            let (_, coefficients) = constant_chart(&rows, omitted)?
                .ok_or_else(|| "certified planar chart is structurally singular".to_string())?;
            let relation = FivePointRelation {
                active: Atom::num(1),
                coefficients: coefficients
                    .into_iter()
                    .enumerate()
                    .map(|(child,c)| if self.certified_planar_mask & (1 << child) != 0 {
                        Atom::new()
                    } else { self.bind_alias(c) })
                    .collect(),
            };
            self.five_point_relations
                .insert(active.to_vec(), Some(relation.clone()));
            return Ok(Some(relation));
        }
        let _determinant = PreparationTimer::new(
            self.profile_preparation,
            format_args!("coordinate determinant {active:?}"),
        );
        let coordinate_determinant = Matrix::from_nested_vec(rows.clone(), field())
            .map_err(|e| format!("PV pentagon coordinate matrix: {e}"))?
            .det()
            .map_err(|e| format!("PV pentagon coordinate determinant: {e:?}"))?;
        drop(_determinant);
        if matches!(coordinate_determinant.as_view(), AtomView::Num(_))
            && !coordinate_determinant.is_zero()
        {
            self.five_point_relations.insert(active.to_vec(), None);
            return Ok(None);
        }
        rows.push(self.f_values(active));
        if coordinate_determinant.together().is_zero() {
            // Structural identities remain exact under every allowed symbolic
            // variation, so their full Dual jets require no primal projection.
            let matrix = Matrix::from_nested_vec(rows, field())
                .map_err(|e| format!("PV pentagon denominator matrix: {e}"))?;
            let rhs = Matrix::new_vec(
                vec![
                    Atom::new(),
                    Atom::new(),
                    Atom::new(),
                    Atom::new(),
                    Atom::num(1),
                ],
                field(),
            );
            let relation = matrix.solve_any(&rhs).ok().map(|solution| {
                let mut coefficients = vec![Atom::new()];
                coefficients.extend((0..4).map(|i| solution[(i, 0)].clone()));
                coefficients[0] = -coefficients[1..].iter().fold(Atom::new(), |sum, c| sum + c);
                FivePointRelation {
                    active: Atom::num(1),
                    coefficients: coefficients
                        .into_iter()
                        .map(|c| self.bind_alias(c))
                        .collect(),
                }
            });
            self.five_point_relations
                .insert(active.to_vec(), relation.clone());
            return Ok(relation);
        }
        if self.analytic_jet_order > 0 {
            let note = "Generic rank-three pentagon pinching is disabled for Dual layouts until every Taylor coefficient of the coordinate-rank constraint is certified; structurally exact rank-three families remain supported";
            if !self.preparation_notes.iter().any(|value| value == note) {
                self.preparation_notes.push(note.into());
            }
            self.five_point_relations.insert(active.to_vec(), None);
            return Ok(None);
        }
        let mut charts = Vec::new();
        for omitted in 0..4 {
            let _chart = PreparationTimer::new(
                self.profile_preparation,
                format_args!("pentagon constant chart {active:?}, omitted row {omitted}"),
            );
            let Some((determinant, coefficients)) = constant_chart(&rows, omitted)? else {
                continue;
            };
            charts.push((self.bind_alias(determinant), coefficients));
        }
        let mut available = Atom::new();
        let mut coefficients = vec![Atom::new(); 5];
        for (determinant, candidate) in charts.into_iter().rev() {
            available = conditional(&determinant, &Atom::num(1), &available);
            for (coefficient, value) in coefficients.iter_mut().zip(candidate) {
                *coefficient = conditional(&determinant, &value, coefficient);
            }
        }
        let relation = if available.is_zero() {
            None
        } else {
            let determinant_alias = self.bind_alias(coordinate_determinant.clone());
            let enabled = conditional(&determinant_alias, &Atom::new(), &available);
            let enabled = self.bind_alias(enabled);
            self.coordinate_constraints.push(CoordinateConstraint {
                owner: enabled.clone(),
                active: enabled.clone(),
                expression: coordinate_determinant,
                label: format!("rank-three pentagon coordinate identity {active:?}"),
            });
            Some(FivePointRelation {
                active: enabled,
                coefficients: coefficients
                    .into_iter()
                    .map(|c| self.bind_alias(c))
                    .collect(),
            })
        };
        self.five_point_relations
            .insert(active.to_vec(), relation.clone());
        Ok(relation)
    }

    pub(crate) fn denominator_identity(&mut self, active: &[usize]) -> Result<Vec<Atom>, String> {
        if let Some(factors) = self.denominator_identities.get(active) {
            return Ok(factors.clone());
        }
        let _identity = PreparationTimer::new(
            self.profile_preparation,
            format_args!("high-point denominator identity {active:?}"),
        );
        let origin = &self.shifts[active[0]];
        let relative: Vec<[Atom; 4]> = active[1..]
            .iter()
            .map(|&i| std::array::from_fn(|mu| &self.shifts[i][mu] - &origin[mu]))
            .collect();
        let mut rows: Vec<Vec<Atom>> = (0..4)
            .map(|mu| relative.iter().map(|p| p[mu].clone()).collect())
            .collect();
        rows.push(self.f_values(active));
        let rhs = Matrix::new_vec(
            vec![
                Atom::new(),
                Atom::new(),
                Atom::new(),
                Atom::new(),
                Atom::num(1),
            ],
            field(),
        );
        // A generic seven-point system has six possible five-column bases.
        // One symbolic pivot choice can vanish on an otherwise regular lane.
        // Keep all bases and select them lazily by their exact determinants.
        // On overlaps the identities agree as functions of q, including Dual
        // jets; changing the algebraic chart is not a physical branch change.
        let symbolic_seven = active.len() == 7
            && rows
                .iter()
                .flatten()
                .any(|a| !matches!(a.as_view(), AtomView::Num(_)));
        if symbolic_seven {
            let mut charts = Vec::new();
            for omitted in (0..6).rev() {
                let _chart = PreparationTimer::new(
                    self.profile_preparation,
                    format_args!("seven-point chart {active:?}, omitted column {omitted}"),
                );
                let columns: Vec<_> = (0..6).filter(|&i| i != omitted).collect();
                let matrix = Matrix::from_nested_vec(
                    rows.iter()
                        .map(|row| columns.iter().map(|&i| row[i].clone()).collect())
                        .collect(),
                    field(),
                )
                .map_err(|e| format!("PV denominator chart: {e}"))?;
                let _determinant = PreparationTimer::new(
                    self.profile_preparation,
                    format_args!("seven-point chart determinant"),
                );
                let determinant = matrix
                    .det()
                    .map_err(|e| format!("PV denominator chart determinant: {e:?}"))?;
                drop(_determinant);
                if determinant.is_zero() {
                    continue;
                }
                let _solve = PreparationTimer::new(
                    self.profile_preparation,
                    format_args!("seven-point chart solve"),
                );
                let solution = matrix
                    .solve(&rhs)
                    .map_err(|e| format!("PV denominator chart solution: {e:?}"))?;
                drop(_solve);
                let mut factors = vec![Atom::new(); active.len()];
                for (i, &column) in columns.iter().enumerate() {
                    factors[column + 1] = solution[(i as u32, 0)].clone();
                }
                factors[0] = -factors[1..]
                    .iter()
                    .fold(Atom::new(), |sum, value| sum + value);
                charts.push((self.bind_alias(determinant), factors));
            }
            if let Some((_, mut factors)) = charts.pop() {
                // If every chart is singular, the final chart still has an
                // active zero denominator. Runtime guards reject that lane.
                // Returning zero here would incorrectly report success.
                for (determinant, candidate) in charts.into_iter().rev() {
                    for (factor, candidate) in factors.iter_mut().zip(candidate) {
                        *factor = conditional(&determinant, &candidate, factor);
                    }
                }
                let factors: Vec<_> = factors
                    .into_iter()
                    .map(|factor| self.bind_alias(factor))
                    .collect();
                self.denominator_identities
                    .insert(active.to_vec(), factors.clone());
                return Ok(factors);
            }
        }
        let matrix = Matrix::from_nested_vec(rows, field())
            .map_err(|e| format!("PV denominator system: {e}"))?;
        let solution = matrix
            .solve_any(&rhs)
            .map_err(|e| format!("PV degenerate denominator identity: {e:?}"))?;
        let mut factors = vec![Atom::new()];
        for i in 0..active.len() - 1 {
            factors.push(solution[(i as u32, 0)].clone());
        }
        factors[0] = -factors[1..]
            .iter()
            .fold(Atom::new(), |sum, value| sum + value);
        self.denominator_identities
            .insert(active.to_vec(), factors.clone());
        Ok(factors)
    }
}

pub(crate) struct PreparationTimer {
    started: Option<Instant>,
    label: String,
}

impl PreparationTimer {
    pub(crate) fn new(enabled: bool, label: std::fmt::Arguments<'_>) -> Self {
        let label = if enabled {
            label.to_string()
        } else {
            String::new()
        };
        if enabled {
            eprintln!("PV prepare begin {label}");
        }
        Self {
            started: enabled.then(Instant::now),
            label,
        }
    }
}

impl Drop for PreparationTimer {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            eprintln!("PV prepare end {}: {:?}", self.label, started.elapsed());
        }
    }
}

fn conditional(condition: &Atom, when_true: &Atom, when_false: &Atom) -> Atom {
    if when_true == when_false {
        when_true.clone()
    } else {
        Symbol::IF.call((condition, when_true, when_false))
    }
}

fn field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: Some(Box::new(|view, output| {
            *output = view.together();
            true
        })),
    }
}

fn dot(p: &[Atom; 4], q: &[Atom; 4]) -> Atom {
    &p[0] * &q[0] - &p[1] * &q[1] - &p[2] * &q[2] - &p[3] * &q[3]
}

fn cartesian_dot(p: &[Atom; 4], q: &[Atom; 4]) -> Atom {
    (0..4).fold(Atom::new(), |sum, mu| sum + &p[mu] * &q[mu])
}

fn without(active: &[usize], index: usize) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (i != index).then_some(x))
        .collect()
}

fn factorial(n: u16) -> Atom {
    Atom::num(Integer::factorial(u32::from(n)))
}
fn binomial(n: u16, k: u16) -> Atom {
    Atom::num(Integer::binom(i64::from(n), i64::from(k)))
}
fn pow(atom: &Atom, exponent: u16) -> Atom {
    if exponent == 0 {
        Atom::num(1)
    } else {
        atom.pow(Atom::num(i64::from(exponent)))
    }
}
fn monomial(atoms: &[Atom], exponents: &[u16]) -> Atom {
    atoms
        .iter()
        .zip(exponents)
        .fold(Atom::num(1), |term, (atom, &n)| term * pow(atom, n))
}

fn compositions(total: u16, count: usize) -> Vec<Vec<u16>> {
    fn visit(total: u16, count: usize, prefix: &mut Vec<u16>, result: &mut Vec<Vec<u16>>) {
        if count == 0 {
            if total == 0 {
                result.push(prefix.clone());
            }
        } else {
            for first in 0..=total {
                prefix.push(first);
                visit(total - first, count - 1, prefix, result);
                prefix.pop();
            }
        }
    }
    let mut result = Vec::new();
    visit(total, count, &mut Vec::new(), &mut result);
    result
}

use crate::ibp::{accumulate, divide_dimension, scale, zero};

/// Raw polynomial predicates for the exact represented-input certificate.
/// The first root is det(K); roots 1..5 are the four constant-one charts.
/// This uses the same chart matrix construction as the PV identity solver.
pub(crate) fn planar_chart_polynomials(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
) -> Result<Vec<Atom>, String> {
    if shifts.len() != 5 || masses.len() != 5 {
        return Err("planar certificate requires five shifts and squared masses".into());
    }
    let active = [0, 1, 2, 3, 4];
    let mut rows = coordinate_rows(shifts, &active);
    let mut expressions = vec![
        Matrix::from_nested_vec(rows.clone(), field())
            .map_err(|e| e.to_string())?
            .det()
            .map_err(|e| format!("planar coordinate determinant: {e:?}"))?,
    ];
    rows.push(
        (1..5)
            .map(|i| {
                let relative = std::array::from_fn(|mu| &shifts[i][mu] - &shifts[0][mu]);
                dot(&relative, &relative) - &masses[i] + &masses[0]
            })
            .collect(),
    );
    for omitted in 0..4 {
        expressions.push(
            chart_matrix(&rows, omitted)?
                .det()
                .map_err(|e| format!("planar constant chart determinant: {e:?}"))?,
        );
    }
    Ok(expressions)
}

/// Raw polynomial Cramer numerators, chart-major then all five child
/// coefficients. They share the determinant already returned by
/// `planar_chart_polynomials`. The origin numerator is the negative sum.
pub(crate) fn planar_chart_cofactor_polynomials(shifts:&[[Atom;4]],masses:&[Atom])->Result<Vec<Atom>,String> {
    if shifts.len()!=5 || masses.len()!=5 {return Err("planar cofactors require five shifts and masses".into());}
    let mut rows=coordinate_rows(shifts,&[0,1,2,3,4]);
    rows.push((1..5).map(|i|{let relative=std::array::from_fn(|mu|&shifts[i][mu]-&shifts[0][mu]);dot(&relative,&relative)-&masses[i]+&masses[0]}).collect());
    let mut result=Vec::with_capacity(20);
    for omitted in 0..4 {
        let chart_rows:Vec<_>=rows.iter().enumerate().filter_map(|(row,x)|(row!=omitted).then_some(x.clone())).collect();
        let mut numerators=vec![Atom::new()];
        for column in 0..4 {
            let replaced=chart_rows.iter().enumerate().map(|(row,entries)|entries.iter().enumerate().map(|(j,x)|
                if j==column {Atom::num(i64::from(row==3))}else{x.clone()}).collect()).collect();
            numerators.push(Matrix::from_nested_vec(replaced,field()).map_err(|e|e.to_string())?.det().map_err(|e|format!("planar cofactor determinant: {e:?}"))?);
        }
        numerators[0]=-Atom::add_many(&numerators[1..]);
        result.extend(numerators);
    }
    Ok(result)
}

fn coordinate_rows(shifts: &[[Atom; 4]], active: &[usize]) -> Vec<Vec<Atom>> {
    (0..4)
        .map(|mu| {
            active[1..]
                .iter()
                .map(|&i| &shifts[i][mu] - &shifts[active[0]][mu])
                .collect()
        })
        .collect()
}

fn chart_matrix(rows: &[Vec<Atom>], omitted: usize) -> Result<Matrix<AtomField>, String> {
    Matrix::from_nested_vec(
        rows.iter()
            .enumerate()
            .filter(|(i, _)| *i != omitted)
            .map(|(_, row)| row.clone())
            .collect(),
        field(),
    )
    .map_err(|e| format!("PV pentagon constant-identity chart: {e}"))
}

fn constant_chart(rows: &[Vec<Atom>], omitted: usize) -> Result<Option<(Atom, Vec<Atom>)>, String> {
    let matrix = chart_matrix(rows, omitted)?;
    let determinant = matrix
        .det()
        .map_err(|e| format!("PV pentagon constant-identity determinant: {e:?}"))?;
    if determinant.is_zero() {
        return Ok(None);
    }
    let rhs = Matrix::new_vec(
        vec![Atom::new(), Atom::new(), Atom::new(), Atom::num(1)],
        field(),
    );
    let solution = matrix
        .solve(&rhs)
        .map_err(|e| format!("PV pentagon constant-identity solve: {e:?}"))?;
    let mut coefficients = vec![Atom::new()];
    coefficients.extend((0..4).map(|i| solution[(i, 0)].clone()));
    coefficients[0] = -coefficients[1..].iter().fold(Atom::new(), |sum, c| sum + c);
    Ok(Some((determinant, coefficients)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_shifts(rows: &[[i32; 4]]) -> Vec<[Atom; 4]> {
        rows.iter().map(|p| p.map(Atom::num)).collect()
    }

    #[test]
    fn generalized_pv_scalar_conversion_matches_independent_ibp_coefficients() {
        let points = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]];
        for (powers, dimension_shift) in [
            (vec![1], 1),
            (vec![1, 2], 1),
            (vec![1, 3], 2),
            (vec![1, 2], 2),
            (vec![1, 2, 3], 4),
            (vec![1, 1, 1, 1], 1),
            (vec![1, 1, 2, 1], 2),
        ] {
            let n = powers.len();
            let shifts = exact_shifts(&points[..n]);
            let masses: Vec<_> = (0..n).map(|i| Atom::num(4 + 3 * i)).collect();
            let r = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let difference =
                                std::array::from_fn(|mu| &shifts[i][mu] - &shifts[j][mu]);
                            dot(&difference, &difference) - &masses[i] - &masses[j]
                        })
                        .collect()
                })
                .collect();
            let mut ibp = ScalarCompiler::new(r).unwrap();
            let mut pv = PvCompiler::new(shifts, masses).unwrap();
            let request = ScalarRequest {
                powers,
                dimension_shift,
            };
            let actual = pv.compile_generalized_scalar(&request).unwrap().unwrap();
            let expected = ibp.compile(request.clone()).unwrap();
            for slot in 0..4 {
                assert!(
                    (&actual[slot] - &expected[slot]).together().is_zero(),
                    "{request:?}, slot {slot}: {}",
                    &actual[slot] - &expected[slot]
                );
            }
        }
    }

    #[test]
    fn generalized_pv_eligibility_and_coefficient_budget_are_explicit() {
        let mut bubble = PvCompiler::new(
            exact_shifts(&[[0, 0, 0, 0], [0, 1, 0, 0]]),
            vec![Atom::num(4), Atom::num(9)],
        )
        .unwrap();
        for (powers, dimension_shift) in [(vec![2, 1], 1), (vec![1, 1], -1), (vec![1, 3], 1)] {
            assert!(
                bubble
                    .compile_generalized_scalar(&ScalarRequest {
                        powers,
                        dimension_shift,
                    })
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            bubble
                .compile_generalized_scalar(&ScalarRequest {
                    powers: vec![1, 1],
                    dimension_shift: 17,
                })
                .unwrap_err()
                .contains("preparation budget")
        );
        let mut pentagon = PvCompiler::new(
            exact_shifts(&[
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ]),
            vec![Atom::num(4); 5],
        )
        .unwrap();
        assert!(
            pentagon
                .compile_generalized_scalar(&ScalarRequest {
                    powers: vec![1; 5],
                    dimension_shift: 1,
                })
                .unwrap_err()
                .contains("independent shifted-master completion")
        );
    }

    #[test]
    fn tadpole_translation_and_metric_rational_term() {
        let mut compiler =
            PvCompiler::new(exact_shifts(&[[1, 0, 0, 0]]), vec![Atom::num(4)]).unwrap();
        let time = compiler.compile_moment([1, 0, 0, 0]).unwrap();
        let xx = compiler.compile_moment([0, 2, 0, 0]).unwrap();
        let leaf = &compiler.leaves()[0];
        for slot in 0..4 {
            assert!((&time[slot] + &leaf.slots[slot]).together().is_zero());
        }
        assert!((&xx[0] + &leaf.slots[0]).together().is_zero());
        assert!(
            (&xx[3]
                + &leaf.slots[3]
                + (&leaf.slots[0] + &leaf.slots[1]) / Atom::num(2)
                + &leaf.slots[2] / Atom::num(4))
            .together()
            .is_zero()
        );
    }

    #[test]
    fn bubble_rank_one_and_uv_tensor_components() {
        let mut compiler = PvCompiler::new(
            exact_shifts(&[[0, 0, 0, 0], [0, 1, 0, 0]]),
            vec![Atom::num(4), Atom::num(9)],
        )
        .unwrap();
        let bx = compiler.compile_moment([0, 1, 0, 0]).unwrap();
        let btt = compiler.compile_moment([2, 0, 0, 0]).unwrap();
        let bxx = compiler.compile_moment([0, 2, 0, 0]).unwrap();
        let mut values: HashMap<Atom, f64> = HashMap::new();
        for leaf in compiler.leaves() {
            for slot in 0..4 {
                let residue = if slot == 0 {
                    match leaf.active.as_slice() {
                        [0] => 4.,
                        [1] => 9.,
                        [0, 1] => 1.,
                        _ => unreachable!(),
                    }
                } else {
                    0.
                };
                values.insert(leaf.slots[slot].clone(), residue);
            }
        }
        assert!((bx[0].evaluate(&values).unwrap() + 0.5).abs() < 1e-14);
        assert!((btt[0].evaluate(&values).unwrap() - 10. / 3.).abs() < 1e-14);
        assert!((bxx[0].evaluate(&values).unwrap() + 3.).abs() < 1e-14);
    }

    #[test]
    fn denominator_identity_is_exact_in_symbolica() {
        let points = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 1, 0, 1],
        ];
        for n in [6, 7] {
            let mut compiler = PvCompiler::new(
                exact_shifts(&points[..n]),
                (0..n).map(|i| Atom::num((i + 2).pow(2) as i64)).collect(),
            )
            .unwrap();
            let active: Vec<_> = (0..n).collect();
            let factors = compiler.denominator_identity(&active).unwrap();
            assert!(
                factors
                    .iter()
                    .fold(Atom::new(), |sum, x| sum + x)
                    .together()
                    .is_zero()
            );
            for mu in 0..4 {
                assert!(
                    factors
                        .iter()
                        .enumerate()
                        .fold(Atom::new(), |sum, (i, c)| sum + c * &compiler.shifts[i][mu])
                        .together()
                        .is_zero()
                );
            }
            let constant = factors.iter().enumerate().fold(Atom::new(), |sum, (i, c)| {
                sum + c
                    * (dot(&compiler.shifts[i], &compiler.shifts[i]) - &compiler.mass_squared[i])
            });
            assert!((constant - Atom::num(1)).together().is_zero());
        }
    }

    #[test]
    fn alias_dag_matches_inline_and_preserves_supplied_distances() {
        let shifts = exact_shifts(&[[1, -2, 3, -1], [1, -1, 3, -1]]);
        let masses = vec![Atom::num(4), Atom::num(9)];
        let mut inline = PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
        let mut dag = PvCompiler::with_aliases(shifts.clone(), masses.clone()).unwrap();
        for monomial in crate::basis::tensor_basis(3) {
            let mut expected = inline.compile_moment(monomial.0).unwrap();
            let physical_poles: Vec<_> = inline
                .leaves()
                .iter()
                .flat_map(|leaf| {
                    let mut replacements = vec![Replacement::new(
                        leaf.slots[0].clone(),
                        match leaf.active.len() {
                            1 => masses[leaf.active[0]].clone(),
                            2 => Atom::num(1),
                            _ => Atom::new(),
                        },
                    )];
                    if leaf.active.len() == 1 {
                        replacements.push(Replacement::new(leaf.slots[1].clone(), Atom::new()));
                        replacements.push(Replacement::new(leaf.slots[2].clone(), Atom::new()));
                    }
                    replacements
                })
                .collect();
            expected = expected.map(|value| value.replace_multiple(physical_poles.iter()));
            let mut actual = dag.compile_moment(monomial.0).unwrap();
            for (handle, body) in dag.aliases().iter().rev() {
                let replacement = Replacement::new(handle.clone(), body.clone());
                actual = actual.map(|value| value.replace_multiple(std::iter::once(&replacement)));
            }
            for slot in 0..4 {
                assert!((&actual[slot] - &expected[slot]).together().is_zero());
            }
        }
        assert!(!dag.aliases().is_empty());
        let distances = vec![
            vec![Atom::new(), Atom::num(-2)],
            vec![Atom::num(-2), Atom::new()],
        ];
        let mut supplied = PvCompiler::with_distances(
            exact_shifts(&[[0, 0, 0, 0], [0, 1, 0, 0]]),
            masses,
            distances,
            false,
        )
        .unwrap();
        let value = supplied.compile_moment([0, 1, 0, 0]).unwrap();
        let mut inputs: HashMap<Atom, f64> = HashMap::new();
        for leaf in supplied.leaves() {
            inputs.insert(
                leaf.slots[3].clone(),
                match leaf.active.as_slice() {
                    [0] => 3.,
                    [1] => 5.,
                    [0, 1] => 7.,
                    _ => unreachable!(),
                },
            );
        }
        assert_eq!(value[3].evaluate(&inputs).unwrap(), -47. / 4.);
    }

    #[test]
    fn lightlike_bubble_branches_match_exact_coefficients() {
        use symbolica::{
            domains::{float::Complex, rational::Rational},
            evaluate::ExpressionEvaluator,
        };
        let s = Atom::var(symbolica::symbol!("iregi::pv_test_s"));
        let m0 = Atom::var(symbolica::symbol!("iregi::pv_test_m0"));
        let m1 = Atom::var(symbolica::symbol!("iregi::pv_test_m1"));
        let shifts = exact_shifts(&[[0, 0, 0, 0], [1, 1, 0, 0]]);
        let distances = vec![vec![Atom::new(), s.clone()], vec![s.clone(), Atom::new()]];
        let mut generic = PvCompiler::with_distances(
            shifts.clone(),
            vec![m0.clone(), m1.clone()],
            distances,
            true,
        )
        .unwrap();
        let basis = crate::basis::tensor_basis(3);
        let expressions: Vec<_> = basis
            .iter()
            .flat_map(|m| generic.compile_moment(m.0).unwrap())
            .collect();
        let mut parameters = vec![s, m0, m1];
        parameters.extend(generic.leaves().iter().flat_map(|leaf| leaf.slots.clone()));
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&expressions, &parameters)
                .add_aliases(generic.aliases().iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        let mut evaluator = exact.map_coeff(&|x| f64::from(x.re.clone()));
        for (distance, a, b) in [(-1, 4, 9), (0, 4, 9), (0, 4, 4), (0, 0, 0)] {
            let mut literal = PvCompiler::with_distances(
                shifts.clone(),
                vec![Atom::num(a), Atom::num(b)],
                vec![
                    vec![Atom::new(), Atom::num(distance)],
                    vec![Atom::num(distance), Atom::new()],
                ],
                true,
            )
            .unwrap();
            let expected: Vec<_> = basis
                .iter()
                .flat_map(|m| literal.compile_moment(m.0).unwrap())
                .collect();
            let mut inputs = vec![f64::from(distance), f64::from(a), f64::from(b)];
            let mut values = HashMap::new();
            for leaf in generic.leaves() {
                for slot in 0..4 {
                    let value = match (leaf.active.as_slice(), slot) {
                        ([0], 0) => f64::from(a),
                        ([1], 0) => f64::from(b),
                        ([0, 1], 0) => 1.,
                        ([0], 3) => {
                            if a == 0 {
                                0.
                            } else {
                                3.
                            }
                        }
                        ([1], 3) => {
                            if b == 0 {
                                0.
                            } else if a == b {
                                3.
                            } else {
                                5.
                            }
                        }
                        ([0, 1], 1) => {
                            if a == 0 && b == 0 && distance == 0 {
                                -1.
                            } else {
                                0.
                            }
                        }
                        ([0, 1], 3) => {
                            if distance == 0 && a == b {
                                if a == 0 { 0. } else { 3. / f64::from(a) - 1. }
                            } else {
                                7.
                            }
                        }
                        _ => 0.,
                    };
                    inputs.push(value);
                    values.insert(leaf.slots[slot].clone(), value);
                }
            }
            let mut outputs = vec![0.; expressions.len()];
            evaluator.evaluate(&inputs, &mut outputs);
            for (i, expression) in expected.into_iter().enumerate() {
                let expanded = literal
                    .aliases()
                    .iter()
                    .rev()
                    .fold(expression, |value, (alias, body)| {
                        value.replace(alias.clone()).with(body.clone())
                    });
                let expected = expanded.evaluate(&values).unwrap();
                assert!(
                    (outputs[i] - expected).abs() < 1e-11 * (1. + expected.abs()),
                    "distance={distance}, masses=({a},{b}), index={i}, actual={}, expected={expected}",
                    outputs[i]
                );
            }
        }
    }

    #[test]
    fn lightlike_moments_match_independent_parameter_integral() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/pv-high-rank/lightlike-feynman-parameters.json"
        ))
        .unwrap();
        let rows = report["precision_checks"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["results"]
            .as_array()
            .unwrap();
        for row in rows {
            let masses: Vec<i64> = row["mass_squared"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m.as_i64().unwrap())
                .collect();
            let rank = row["rank"].as_u64().unwrap() as u16;
            let mut compiler = PvCompiler::new(
                exact_shifts(&[[0, 0, 0, 0], [1, 1, 0, 0]]),
                masses.iter().copied().map(Atom::num).collect(),
            )
            .unwrap();
            let value = compiler.compile_moment([0, rank, 0, 0]).unwrap();
            let scalar = rows
                .iter()
                .find(|r| r["rank"] == 0 && r["mass_squared"] == row["mass_squared"])
                .unwrap()["oneloop"]["finite"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let mut inputs = HashMap::new();
            for leaf in compiler.leaves() {
                for slot in 0..4 {
                    let input = match (leaf.active.as_slice(), slot) {
                        ([i], 0) => masses[*i] as f64,
                        ([i], 3) => {
                            let mass = masses[*i] as f64;
                            mass * (1. - (mass / 9.).ln())
                        }
                        ([0, 1], 0) => 1.,
                        ([0, 1], 3) => scalar,
                        _ => 0.,
                    };
                    inputs.insert(leaf.slots[slot].clone(), input);
                }
            }
            for (slot, name) in ["uv", "ir", "ir2", "finite"].into_iter().enumerate() {
                let expected = row["oneloop"][name]
                    .as_str()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap();
                let actual = value[slot].evaluate(&inputs).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-12 * (1. + expected.abs()),
                    "masses={masses:?} rank={rank} {name}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn lightlike_dual_coefficients_match_parameter_derivatives() {
        use symbolica::{
            domains::{dual::HyperDual, float::Complex, rational::Rational},
            evaluate::{Dualizer, ExpressionEvaluator},
        };
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/pv-high-rank/lightlike-feynman-parameters.json"
        ))
        .unwrap();
        let rows = report["precision_checks"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["results"]
            .as_array()
            .unwrap();
        for masses in [[4_i64, 9_i64], [4, 4]] {
            let s = Atom::var(symbolica::symbol!("iregi::pv_test_dual_s"));
            let mut compiler = PvCompiler::with_distances(
                exact_shifts(&[[0, 0, 0, 0], [1, 1, 0, 0]]),
                masses.map(Atom::num).to_vec(),
                vec![vec![Atom::new(), s.clone()], vec![s.clone(), Atom::new()]],
                true,
            )
            .unwrap();
            compiler.set_analytic_jet_order(1).unwrap();
            let expressions: Vec<_> = (0..=3)
                .flat_map(|rank| compiler.compile_moment([0, rank, 0, 0]).unwrap())
                .collect();
            assert!(compiler.set_analytic_jet_order(2).is_err());
            let mut parameters = vec![s];
            parameters.extend(compiler.leaves().iter().flat_map(|leaf| leaf.slots.clone()));
            let exact: ExpressionEvaluator<Complex<Rational>> =
                Atom::evaluator_multiple(&expressions, &parameters)
                    .add_aliases(compiler.aliases().iter().cloned())
                    .unwrap()
                    .direct_translation(true)
                    .horner_iterations(0)
                    .cores(1)
                    .build()
                    .unwrap();
            let exact = exact
                .vectorize(&Dualizer::new(
                    HyperDual::<f64>::new(vec![vec![0], vec![1]]),
                    vec![],
                ))
                .unwrap();
            let mut evaluator = exact.map_coeff(&|x| f64::from(x.re.clone()));
            let matching: Vec<_> = rows
                .iter()
                .filter(|row| {
                    row["mass_squared"][0] == masses[0] && row["mass_squared"][1] == masses[1]
                })
                .collect();
            let scalar = matching.iter().find(|row| row["rank"] == 0).unwrap();
            let mut inputs = vec![0., 1.];
            for leaf in compiler.leaves() {
                for slot in 0..4 {
                    let values = match (leaf.active.as_slice(), slot) {
                        ([i], 0) => [masses[*i] as f64, 0.],
                        ([i], 3) => {
                            let mass = masses[*i] as f64;
                            [mass * (1. - (mass / 9.).ln()), 0.]
                        }
                        ([0, 1], 0) => [1., 0.],
                        ([0, 1], 3) => [
                            scalar["oneloop"]["finite"]
                                .as_str()
                                .unwrap()
                                .parse()
                                .unwrap(),
                            scalar["derivative_invariant_s"]["oneloop"]["finite"]
                                .as_str()
                                .unwrap()
                                .parse()
                                .unwrap(),
                        ],
                        _ => [0., 0.],
                    };
                    inputs.extend(values);
                }
            }
            let mut outputs = vec![0.; expressions.len() * 2];
            evaluator.evaluate(&inputs, &mut outputs);
            for row in matching {
                let rank = row["rank"].as_u64().unwrap() as usize;
                for (slot, name) in ["uv", "ir", "ir2", "finite"].into_iter().enumerate() {
                    for component in 0..2 {
                        let expected = if component == 0 {
                            &row["oneloop"][name]
                        } else {
                            &row["derivative_invariant_s"]["oneloop"][name]
                        }
                        .as_str()
                        .unwrap()
                        .parse::<f64>()
                        .unwrap();
                        let actual = outputs[(rank * 4 + slot) * 2 + component];
                        assert!(
                            (actual - expected).abs() < 1e-12 * (1. + expected.abs()),
                            "masses={masses:?} rank={rank} {name} component={component}: actual={actual}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn seven_point_chart_switch_preserves_identity_and_dual_jet() {
        use symbolica::{
            domains::{dual::HyperDual, float::Complex, rational::Rational},
            evaluate::{Dualizer, ExpressionEvaluator},
        };
        let x = Atom::var(symbolica::symbol!("iregi::pv_chart_mass_offset"));
        let shifts = exact_shifts(&[
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 3, 5, 7],
        ]);
        let mut masses: Vec<_> = [4, 9, 16, 25, 36, 74, 64]
            .into_iter()
            .map(Atom::num)
            .collect();
        masses[5] = &masses[5] + &x;
        let mut compiler = PvCompiler::with_aliases(shifts.clone(), masses.clone()).unwrap();
        let mut first_rows: Vec<Vec<Atom>> = (0..4)
            .map(|mu| shifts[1..6].iter().map(|p| p[mu].clone()).collect())
            .collect();
        first_rows.push(compiler.f_values(&[0, 1, 2, 3, 4, 5]));
        let first_determinant = Matrix::from_nested_vec(first_rows, field())
            .unwrap()
            .det()
            .unwrap();
        assert!(
            first_determinant
                .replace(x.clone())
                .with(Atom::new())
                .is_zero()
        );
        assert!(
            !first_determinant
                .replace(x.clone())
                .with(Atom::num(1))
                .is_zero()
        );
        let factors = compiler
            .denominator_identity(&(0..7).collect::<Vec<_>>())
            .unwrap();
        // At x=0, the first five columns have a homogeneous denominator
        // relation, so their chart is singular and cannot produce 1. The last
        // propagator restores the rank and must enter an alternative chart.
        let mut expressions = factors.clone();
        expressions.push(factors.iter().fold(Atom::new(), |sum, c| sum + c));
        for mu in 0..4 {
            expressions.push((0..7).fold(Atom::new(), |sum, i| sum + &factors[i] * &shifts[i][mu]));
        }
        expressions.push((0..7).fold(Atom::new(), |sum, i| {
            sum + &factors[i] * (dot(&shifts[i], &shifts[i]) - &masses[i])
        }));
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&expressions, &[x])
                .add_aliases(compiler.aliases().iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        let exact = exact
            .vectorize(&Dualizer::new(
                HyperDual::<f64>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap();
        let mut evaluator = exact.map_coeff(&|a| f64::from(a.re.clone()));
        for mass_offset in [0., 1., -2.] {
            let mut output = vec![0.; expressions.len() * 2];
            evaluator.evaluate(&[mass_offset, 1.], &mut output);
            assert!(output.iter().all(|value| value.is_finite()));
            if mass_offset == 0. {
                assert!(output[6 * 2].abs() > 1e-5, "alternate chart was not used");
            }
            for identity in 7..expressions.len() {
                let expected = if identity + 1 == expressions.len() {
                    1.
                } else {
                    0.
                };
                assert!((output[identity * 2] - expected).abs() < 1e-12);
                assert!(output[identity * 2 + 1].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn planar_pentagon_all_rank_seven_moments_obey_independent_box_identity() {
        let shifts = exact_shifts(&[
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
        ]);
        let masses = [4, 9, 16, 25, 36].into_iter().map(Atom::num).collect();
        let mut compiler = PvCompiler::new(shifts, masses).unwrap();
        let active = [0, 1, 2, 3, 4];
        // Independently expanding these particular denominators gives
        // 2 D0 - D1 - D2 - D3 + D4 = 6, for every loop momentum.
        let weights = [2, -1, -1, -1, 1];
        for monomial in crate::basis::tensor_basis(7) {
            let actual = compiler.compile_moment(monomial.0).unwrap();
            let mut expected = zero();
            for (i, weight) in weights.into_iter().enumerate() {
                let box_moment = compiler.moment(&without(&active, i), monomial.0).unwrap();
                accumulate(
                    &mut expected,
                    &(Atom::num(weight) / Atom::num(6)),
                    &box_moment,
                );
            }
            for slot in 0..4 {
                assert!(
                    (&actual[slot] - &expected[slot]).together().is_zero(),
                    "{monomial:?}, slot {slot}"
                );
            }
        }
        assert!(compiler.leaves().iter().all(|leaf| leaf.active.len() <= 4));
    }

    #[test]
    fn generic_planar_pentagon_branch_is_lazy_and_preserves_ordinary_points() {
        use symbolica::{
            domains::{float::Complex, rational::Rational},
            evaluate::ExpressionEvaluator,
        };
        let x = Atom::var(symbolica::symbol!("iregi::pv_planar_departure"));
        let points = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
        ];
        let mut shifts = exact_shifts(&points);
        shifts[4][0] = x.clone();
        let masses: Vec<_> = [4, 9, 16, 25, 36].into_iter().map(Atom::num).collect();
        let basis = crate::basis::tensor_basis(2);
        let mut compiler = PvCompiler::with_aliases(shifts.clone(), masses.clone()).unwrap();
        let expressions: Vec<_> = basis
            .iter()
            .flat_map(|m| compiler.compile_moment(m.0).unwrap())
            .collect();
        let mut parameters = vec![x.clone()];
        parameters.extend(
            compiler
                .leaves()
                .iter()
                .flat_map(|leaf| leaf.slots.iter().cloned()),
        );
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&expressions, &parameters)
                .add_aliases(compiler.aliases().iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        let mut evaluator = exact.map_coeff(&|value| {
            Complex::new(f64::from(value.re.clone()), f64::from(value.im.clone()))
        });
        let mut inputs: Vec<_> = (0..parameters.len())
            .map(|i| Complex::new(1. / (i as f64 + 1.), 1. / (i as f64 + 3.)))
            .collect();
        // At +/-1, a three-vector box pinch has a null Gram matrix. Use
        // +/-2 to compare against ordinary points with regular descendants.
        for departure in [0, 2, -2] {
            inputs[0] = Complex::new(f64::from(departure), 0.);
            let mut exact_points = points;
            exact_points[4][0] = departure;
            let mut independent =
                PvCompiler::with_aliases(exact_shifts(&exact_points), masses.clone()).unwrap();
            let roots: Vec<_> = basis
                .iter()
                .flat_map(|m| independent.compile_moment(m.0).unwrap())
                .collect();
            let expected: ExpressionEvaluator<Complex<Rational>> =
                Atom::evaluator_multiple(&roots, &parameters)
                    .add_aliases(independent.aliases().iter().cloned())
                    .unwrap()
                    .direct_translation(true)
                    .horner_iterations(0)
                    .cores(1)
                    .build()
                    .unwrap();
            let mut expected = expected.map_coeff(&|value| {
                Complex::new(f64::from(value.re.clone()), f64::from(value.im.clone()))
            });
            let mut actual = vec![Complex::new(0., 0.); roots.len()];
            let mut reference = actual.clone();
            evaluator.evaluate(&inputs, &mut actual);
            expected.evaluate(&inputs, &mut reference);
            for (slot, (a, b)) in actual.iter().zip(reference).enumerate() {
                assert!(
                    a.re.is_finite() && a.im.is_finite(),
                    "departure {departure}, slot {slot}"
                );
                assert!(
                    (a.re - b.re).abs() < 1e-10 * (1. + b.re.abs())
                        && (a.im - b.im).abs() < 1e-10 * (1. + b.im.abs()),
                    "departure {departure}, slot {slot}: {a:?} vs {b:?}"
                );
            }
        }
        let mut dual = PvCompiler::with_aliases(shifts, masses).unwrap();
        dual.set_analytic_jet_order(1).unwrap();
        assert!(
            dual.five_point_constant_relation(&[0, 1, 2, 3, 4])
                .unwrap()
                .is_none()
        );
        assert!(
            dual.preparation_notes()
                .iter()
                .any(|note| note.contains("every Taylor coefficient"))
        );
    }

    #[test]
    fn gram_transform_uses_raw_products_at_each_origin() {
        use std::{cell::RefCell, rc::Rc};
        let mut compiler = PvCompiler::with_distances(
            exact_shifts(&[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 2, 0]]),
            vec![Atom::num(4); 3],
            vec![
                vec![Atom::new(), Atom::num(-1), Atom::num(-1)],
                vec![Atom::num(-1), Atom::new(), Atom::num(-1)],
                vec![Atom::num(-1), Atom::num(-1), Atom::new()],
            ],
            false,
        )
        .unwrap();
        let products = Rc::new(RefCell::new(Vec::new()));
        let captured = products.clone();
        compiler
            .set_gram_scalar_product_transform(move |raw| {
                captured.borrow_mut().push(raw.clone());
                raw + Atom::num(1)
            })
            .unwrap();
        compiler.geometry(&[0, 1, 2]).unwrap();
        let changed_origin = compiler.geometry(&[1, 2]).unwrap();
        assert_eq!(*products.borrow(), [-1, 0, 0, -4, -5].map(Atom::num));
        assert!(
            (&changed_origin.inverse_z[0][0] + Atom::num(1) / Atom::num(8))
                .together()
                .is_zero()
        );
        assert!(compiler.set_gram_scalar_product_transform(|a| a).is_err());
    }

    #[test]
    fn singular_six_point_subspace_is_an_explicit_error() {
        // Retained from the initial oracle corpus: a null transverse subspace
        // makes a four-denominator pinch Gram matrix singular. Pure PV must
        // report this; the runtime Hybrid strategy owns exceptional fallback.
        let shifts = exact_shifts(&[
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ]);
        let masses = (0..6).map(|i| Atom::num((i + 2) * (i + 2))).collect();
        let mut compiler = PvCompiler::new(shifts, masses).unwrap();
        let error = compiler.compile_moment([2, 0, 0, 0]).unwrap_err();
        assert!(error.contains("singular Gram matrix"), "{error}");
    }

    /// Independent validation: scalar leaf values and complete tensor moments
    /// come from separately executed original IREGI, not from our IBP compiler.
    #[test]
    #[ignore = "requires scripts/build-reference.sh"]
    fn original_iregi_cartesian_moments() {
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let base_points = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [2, 3, 5, 7],
            [11, 13, 17, 19],
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut failures = Vec::new();
        let selected_n: Vec<usize> = std::env::var("IREGI_PV_ORACLE_N")
            .map(|s| {
                s.split(',')
                    .map(|v| v.parse().expect("IREGI_PV_ORACLE_N must contain integers"))
                    .collect()
            })
            .unwrap_or_else(|_| (1..=7).collect());
        assert!(selected_n.iter().all(|&n| (1..=7).contains(&n)));
        let variants: Vec<String> = std::env::var("IREGI_PV_ORACLE_VARIANT")
            .unwrap_or_else(|_| "base,translated,order_reversed,loop_reversed".into())
            .split(',')
            .map(str::to_owned)
            .collect();
        assert!(variants.iter().all(|v| {
            ["base", "translated", "order_reversed", "loop_reversed"].contains(&v.as_str())
        }));
        let max_rank = std::env::var("IREGI_PV_ORACLE_RANK")
            .map(|v| v.parse::<u16>().expect("rank must be an integer"))
            .unwrap_or(7);
        assert!(max_rank <= 7);
        let output_dir = root.join("validation/pv-high-rank/oracle-runs");
        std::fs::create_dir_all(&output_dir).unwrap();
        for (n, variant) in selected_n
            .into_iter()
            .flat_map(|n| variants.iter().map(move |variant| (n, variant)))
        {
            let translation = if variant == "translated" {
                [1, -2, 3, -1]
            } else {
                [0, 0, 0, 0]
            };
            let mut points: Vec<_> = base_points[..n]
                .iter()
                .map(|p| std::array::from_fn(|mu| p[mu] + translation[mu]))
                .collect();
            let mut masses: Vec<i32> = (0..n).map(|i| (i as i32 + 2).pow(2)).collect();
            if variant == "order_reversed" {
                points.reverse();
                masses.reverse();
            }
            if variant == "loop_reversed" {
                for p in &mut points {
                    for component in p {
                        *component = -*component;
                    }
                }
            }
            let record_path = output_dir.join(format!("n{n}-{variant}-rank{max_rank}.json"));
            let mut record = serde_json::json!({"propagators":n,"variant":variant,"rank":max_rank,"shifts":points,"mass_squared":masses,"mu":3,"tolerance":"1e-10 mixed after dimensional normalization","status":"started",
                "rust_evaluation":"Exact numeric-specialized Symbolica PV coefficients evaluated in binary64 with original Fortran scalar master leaves; not the public Rust Workspace or native Rust masters",
                "reference_evaluation":"Original IREGI IBP, complete tensor array",
                "normalization":"Native IREGI; UV, IR single, IR double, finite"});
            std::fs::write(
                &record_path,
                serde_json::to_string_pretty(&record).unwrap() + "\n",
            )
            .unwrap();
            eprintln!("PV oracle preparing n={n} variant={variant} rank={max_rank}");
            let mut compiler = PvCompiler::new(
                exact_shifts(&points[..n]),
                masses.iter().copied().map(Atom::num).collect(),
            )
            .unwrap();
            let basis = crate::basis::tensor_basis(max_rank);
            let compiled = basis
                .iter()
                .enumerate()
                .map(|(index, m)| {
                    if index == 0 || basis[index - 1].rank() != m.rank() {
                        eprintln!("PV oracle n={n} variant={variant}: preparing rank {} at moment {index}/{}", m.rank(), basis.len());
                    }
                    compiler.compile_moment(m.0)
                })
                .collect::<Result<Vec<_>, _>>();
            let compiled = match compiled {
                Ok(compiled) => compiled,
                Err(error) => {
                    let failure = format!("n={n} variant={variant}: preparation failed: {error}");
                    failures.push(failure.clone());
                    record["status"] = serde_json::json!("preparation_failed");
                    record["failure"] = serde_json::json!(failure);
                    std::fs::write(
                        &record_path,
                        serde_json::to_string_pretty(&record).unwrap() + "\n",
                    )
                    .unwrap();
                    continue;
                }
            };
            let mut input = String::new();
            let mut request = |id: &str, active: &[usize], rank: u16| {
                input += &format!(
                    "EVAL {id} tensor {} {} {rank} 0 3 0 1e-6\n",
                    if id == "target" { 0 } else { 1 },
                    active.len()
                );
                for &i in active {
                    let p = points[i];
                    input += &format!("{} {} {} {} {} 0 1\n", p[0], p[1], p[2], p[3], masses[i]);
                }
            };
            request("target", &(0..n).collect::<Vec<_>>(), max_rank);
            for (i, leaf) in compiler.leaves().iter().enumerate() {
                request(&format!("leaf{i}"), &leaf.active, 0);
            }
            input += "QUIT\n";
            let mut process = Command::new(root.join("target/reference/iregi-reference-original"))
                .current_dir(&root)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            // The rank-seven response can exceed the OS pipe capacity.
            // Feed requests while wait_with_output concurrently drains stdout.
            let mut stdin = process.stdin.take().unwrap();
            let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
            let output = process.wait_with_output().unwrap();
            writer.join().unwrap().unwrap();
            assert!(output.status.success());
            let results: HashMap<String, serde_json::Value> = String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .filter_map(|line| line.strip_prefix("RESULT "))
                .map(|json| {
                    let value: serde_json::Value = serde_json::from_str(json).unwrap();
                    (value["id"].as_str().unwrap().to_owned(), value)
                })
                .collect();
            eprintln!(
                "PV oracle n={n} rank={max_rank} variant={variant}: {} master leaves",
                compiler.leaves().len()
            );
            assert_eq!(results.len(), compiler.leaves().len() + 1);
            record["reference"] = results["target"].clone();
            if results["target"]["stable"] != true {
                record["status"] = serde_json::json!("reference_unstable");
                std::fs::write(
                    &record_path,
                    serde_json::to_string_pretty(&record).unwrap() + "\n",
                )
                .unwrap();
                failures.push(format!(
                    "n={n} variant={variant}: original IBP reports instability"
                ));
                continue;
            }
            // Compare dimensionless values at the unchanged 1e-10 mixed
            // tolerance. A rank-r N-point tensor has mass dimension r+4-2N.
            let mut scale_squared = masses
                .iter()
                .map(|&m| f64::from(m).abs())
                .fold(1., f64::max);
            for p in &points[..n] {
                for q in &points[..n] {
                    let d = std::array::from_fn::<_, 4, _>(|mu| f64::from(p[mu] - q[mu]));
                    scale_squared = scale_squared
                        .max((d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]).abs());
                }
            }
            let failures_before = failures.len();
            let mut max_scaled_error = 0.0_f64;
            let mut discrepancy_rows = Vec::new();
            let mut actual_moments: Vec<[[String; 2]; 4]> = (0..basis.len())
                .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| String::new())))
                .collect();
            for component in 0..2 {
                let mut values: HashMap<Atom, f64> = HashMap::new();
                for (i, leaf) in compiler.leaves().iter().enumerate() {
                    let result = &results[&format!("leaf{i}")];
                    assert_eq!(result["stable"], true);
                    for slot in 0..4 {
                        values.insert(
                            leaf.slots[slot].clone(),
                            result["moments"][0][slot][component].as_f64().unwrap(),
                        );
                    }
                }
                for (i, value) in compiled.iter().enumerate() {
                    let dimension = basis[i].rank() as i32 + 4 - 2 * n as i32;
                    let unit = scale_squared.sqrt().powi(dimension);
                    for slot in 0..4 {
                        let actual = value[slot].evaluate(&values).unwrap() / unit;
                        let expected = results["target"]["moments"][i][slot][component]
                            .as_f64()
                            .unwrap()
                            / unit;
                        actual_moments[i][slot][component] = format!("{actual:.17e}");
                        let scaled_error = (actual - expected).abs() / (1. + expected.abs());
                        max_scaled_error = max_scaled_error.max(scaled_error);
                        if !actual.is_finite()
                            || !expected.is_finite()
                            || !scaled_error.is_finite()
                            || scaled_error >= 1e-10
                        {
                            failures.push(format!(
                                "n={n} variant={variant} monomial={:?} slot={slot} component={component} normalized actual={actual:.16e} expected={expected:.16e}",basis[i]));
                            discrepancy_rows.push(serde_json::json!({"moment":i,"monomial":basis[i].0,"slot":slot,"component":component,"actual_normalized":format!("{actual:.17e}"),"expected_normalized":format!("{expected:.17e}"),"mixed_error":format!("{scaled_error:.17e}")}));
                        }
                    }
                }
            }
            record["status"] = serde_json::json!(if discrepancy_rows.is_empty() {
                "passed"
            } else {
                "failed"
            });
            record["moments"] = serde_json::json!(basis.len());
            record["master_leaves"] = serde_json::json!(compiler.leaves().len());
            record["rust_original_master_moments_normalized"] = serde_json::json!(actual_moments);
            record["max_mixed_error"] = serde_json::json!(format!("{max_scaled_error:.17e}"));
            record["discrepancies"] = serde_json::json!(discrepancy_rows);
            std::fs::write(
                &record_path,
                serde_json::to_string_pretty(&record).unwrap() + "\n",
            )
            .unwrap();
            eprintln!(
                "PV oracle n={n} variant={variant}: {} failures, maximum normalized mixed error {max_scaled_error:.8e}",
                failures.len() - failures_before
            );
        }
        assert!(
            failures.is_empty(),
            "{} failures; first twenty: {:#?}",
            failures.len(),
            &failures[..failures.len().min(20)]
        );
    }
}

#[cfg(test)]
mod planar_mask_polynomial_tests {
    use super::*;
    #[test]
    fn all_chart_cramer_numerators_match_existing_solve() {
        let x=Atom::var(symbolica::symbol!("planar_mask::x"));
        let y=Atom::var(symbolica::symbol!("planar_mask::y"));
        for variant in 0..6 {
            let mut shifts:Vec<[Atom;4]>=[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]].iter().map(|p|p.map(Atom::num)).collect();
            let mut masses:Vec<_>=[1,3,5,7,11].into_iter().map(Atom::num).collect();
            match variant {
                1=>for p in &mut shifts {for (mu,value) in p.iter_mut().enumerate(){*value=&*value+Atom::num((mu+1) as i64)/Atom::num(3);}},
                2=>{shifts.reverse();masses.reverse();},
                3=>for (i,mass) in masses.iter_mut().enumerate(){*mass=&*mass-Atom::i()*Atom::num((i+1) as i64)/Atom::num(16);},
                4=>shifts[4][3]=&shifts[4][3]+&x*&y,
                5=>for p in &mut shifts {for value in p {*value=-&*value;}},
                _=>(),
            }
            let numerators=planar_chart_cofactor_polynomials(&shifts,&masses).unwrap();
            let determinants=planar_chart_polynomials(&shifts,&masses).unwrap();
            assert_eq!(numerators.len(),20);
            let mut rows=coordinate_rows(&shifts,&[0,1,2,3,4]);
            rows.push((1..5).map(|i|{let p=std::array::from_fn(|mu|&shifts[i][mu]-&shifts[0][mu]);dot(&p,&p)-&masses[i]+&masses[0]}).collect());
            for omitted in 0..4 {
                let (det,coefficients)=constant_chart(&rows,omitted).unwrap().expect("all selected test charts regular");
                assert!((&det-&determinants[omitted+1]).together().is_zero());
                for child in 0..5 {assert!((&det*&coefficients[child]-&numerators[omitted*5+child]).together().is_zero(),"variant={variant} chart={omitted} child={child}");}
            }
        }
    }
    #[test]
    fn private_mask_rejects_invalid_bits_and_omits_child_before_recursion() {
        let shifts:Vec<[Atom;4]>=(0..5).map(|i|std::array::from_fn(|mu|Atom::var(symbolica::symbol!(format!("planar_mask::p{i}_{mu}"))))).collect();
        let masses:Vec<_>=(0..5).map(|i|Atom::var(symbolica::symbol!(format!("planar_mask::m{i}")))).collect();
        let mut compiler=PvCompiler::with_aliases(shifts,masses).unwrap();
        assert!(compiler.set_certified_planar_mask(8).is_err());
        compiler.set_certified_planar_chart(3).unwrap();
        for mask in [31,32,128,255] {assert!(compiler.set_certified_planar_mask(mask).is_err());}
        compiler.set_certified_planar_mask(8).unwrap();
        compiler.compile_moment([1,0,0,0]).unwrap();
        assert!(!compiler.moments.contains_key(&(vec![0,1,2,4],[1,0,0,0])));
        for omitted in [0,1,2,4] {
            let subset:Vec<_>=(0..5).filter(|&i|i!=omitted).collect();
            assert!(compiler.moments.contains_key(&(subset,[1,0,0,0])));
        }
        assert!(compiler.set_certified_planar_mask(0).is_err(),"policy cannot change after compilation");
    }
}
