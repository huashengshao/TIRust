//! Passarino--Veltman tensor recurrences compiled with Symbolica.
//!
//! For up to four denominators the coefficients obey Denner--Dittmaier
//! Eqs. (5.10)--(5.11), including the epsilon-dependent trace denominator.
//! Five-point Cartesian moments use the complete four-dimensional external
//! basis directly; six/seven-point moments use exact denominator identities.
//! This avoids introducing a dimension-shifted pentagon divided by epsilon.
//! Symbolica owns matrix solution, expression arithmetic and polynomial
//! collection. Integer compositions below enumerate tensor indices only.

use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::{atom::AtomField, integer::Integer},
    id::Replacement,
    tensors::matrix::Matrix,
};

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
    aliases: Vec<(Atom, Atom)>,
    scalar: ScalarCompiler,
    bubble_expansions: HashMap<(Vec<usize>, bool), ScalarCompiler>,
    analytic_jet_order: u16,
    coefficients: HashMap<CoefficientKey, [Atom; 4]>,
    geometry: HashMap<Vec<usize>, Geometry>,
    moments: HashMap<(Vec<usize>, [u16; 4]), [Atom; 4]>,
    generated: HashMap<(Vec<usize>, u16), HashMap<Atom, [Atom; 4]>>,
    denominator_identities: HashMap<Vec<usize>, Vec<Atom>>,
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
        Self::with_distances(shifts, mass_squared, distances, alias_enabled)
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
            aliases,
            scalar,
            bubble_expansions: HashMap::new(),
            analytic_jet_order: 0,
            coefficients: HashMap::new(),
            geometry: HashMap::new(),
            moments: HashMap::new(),
            generated: HashMap::new(),
            denominator_identities: HashMap::new(),
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
            values.map(|body| {
                if matches!(body.as_view(), AtomView::Num(_) | AtomView::Var(_)) {
                    return body;
                }
                let alias = Atom::var(symbolica::symbol!(format!(
                    "iregi::pv_node_{}",
                    self.aliases.len()
                )));
                self.aliases.push((alias.clone(), body));
                alias
            })
        } else if self.numeric_kinematics {
            // Exact numeric coefficients collapse to a short master-linear sum.
            values.map(|value| value.expand())
        } else {
            values
        }
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
        } else if rank == 0 {
            self.scalar(active)?
        } else if active.len() == 5 {
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

    /// Generate the polynomial integral of `(z_mu q^mu)^rank` and let
    /// Symbolica collect its four Cartesian indeterminates.
    fn generate_rank(&mut self, active: &[usize], rank: u16) -> Result<(), String> {
        let key = (active.to_vec(), rank);
        if self.generated.contains_key(&key) {
            return Ok(());
        }
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
        let rows = active[1..]
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
            .collect();
        let matrix =
            Matrix::from_nested_vec(rows, field()).map_err(|e| format!("PV Gram matrix: {e}"))?;
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

    pub(crate) fn denominator_identity(&mut self, active: &[usize]) -> Result<Vec<Atom>, String> {
        if let Some(factors) = self.denominator_identities.get(active) {
            return Ok(factors.clone());
        }
        let origin = &self.shifts[active[0]];
        let relative: Vec<[Atom; 4]> = active[1..]
            .iter()
            .map(|&i| std::array::from_fn(|mu| &self.shifts[i][mu] - &origin[mu]))
            .collect();
        let mut rows: Vec<Vec<Atom>> = (0..4)
            .map(|mu| relative.iter().map(|p| p[mu].clone()).collect())
            .collect();
        rows.push(self.f_values(active));
        let matrix = Matrix::from_nested_vec(rows, field())
            .map_err(|e| format!("PV denominator system: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_shifts(rows: &[[i32; 4]]) -> Vec<[Atom; 4]> {
        rows.iter().map(|p| p.map(Atom::num)).collect()
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
        for (n, translation) in (1..=7).map(|n| (n, [0, 0, 0, 0])).chain([
            (2, [1, -2, 3, -1]),
            (3, [1, -2, 3, -1]),
            (5, [1, -2, 3, -1]),
        ]) {
            let points = base_points.map(|p| std::array::from_fn(|mu| p[mu] + translation[mu]));
            let max_rank = 7;
            let masses: Vec<i32> = (0..n).map(|i| (i as i32 + 2).pow(2)).collect();
            let mut compiler = PvCompiler::new(
                exact_shifts(&points[..n]),
                masses.iter().copied().map(Atom::num).collect(),
            )
            .unwrap();
            let basis = crate::basis::tensor_basis(max_rank);
            let compiled = basis
                .iter()
                .map(|m| compiler.compile_moment(m.0))
                .collect::<Result<Vec<_>, _>>();
            let compiled = match compiled {
                Ok(compiled) => compiled,
                Err(error) => {
                    failures.push(format!(
                        "n={n} translation={translation:?}: preparation failed: {error}"
                    ));
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
                "PV oracle n={n} rank={max_rank} translation={translation:?}: {} master leaves",
                compiler.leaves().len()
            );
            assert_eq!(results.len(), compiler.leaves().len() + 1);
            assert_eq!(results["target"]["stable"], true, "reference topology {n}");
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
                        let scaled_error = (actual - expected).abs() / (1. + expected.abs());
                        max_scaled_error = max_scaled_error.max(scaled_error);
                        if !actual.is_finite() || scaled_error >= 1e-10 {
                            failures.push(format!(
                                "n={n} translation={translation:?} monomial={:?} slot={slot} component={component} normalized actual={actual:.16e} expected={expected:.16e}",basis[i]));
                        }
                    }
                }
            }
            eprintln!(
                "PV oracle n={n} translation={translation:?}: {} failures, maximum normalized mixed error {max_scaled_error:.8e}",
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
