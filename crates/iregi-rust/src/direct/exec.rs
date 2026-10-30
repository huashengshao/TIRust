//! Numerical execution of a direct plan.
//!
//! Lanes are processed in three stages. All master arguments of a batch are
//! collected first and each scalar family is evaluated once across lanes. The
//! recurrence program and the Cartesian gather then run per lane on flat,
//! reused buffers. A lane whose required Gram, Cayley or thin-six system is
//! numerically singular reports `Unstable` and exposes no coefficients.

use std::{sync::Arc, time::Instant};

use iregi_masters::{MasterEvaluator, MasterExecutionPolicy, NativeFloat, ScalarIntegral};
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use super::{
    numeric::{
        C, L4, cadd, cfma, cmul, cnorm2, cscale, csub, czero, div_dimension, lfma, lzero,
        solve_in_place,
    },
    plan::{DirectPlanData, NO_GEOMETRY, PostOp, SubsetKind},
};
use crate::{
    BatchResult, Convention, KinematicsBatch, LaneStatus, ReductionError, TensorWeights,
    types::ResultMetadata,
};

/// Stage timings of the most recent profiled call.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirectProfile {
    pub geometry: std::time::Duration,
    pub masters: std::time::Duration,
    pub recurrence: std::time::Duration,
    pub gather: std::time::Duration,
    pub output: std::time::Duration,
}

pub(crate) struct DirectExecutor<T: NativeFloat> {
    plan: Arc<DirectPlanData>,
    prototype: T,
    zero: C<T>,
    one: C<T>,
    constants: Vec<T>,
    evaluators: [Option<MasterEvaluator<T>>; 4],
    master_inputs: [Vec<C<T>>; 4],
    /// Per lane, per family, per master: four Laurent slots.
    master_values: Vec<L4<T>>,
    master_errors: Vec<Option<String>>,
    master_offsets: [usize; 4],
    master_count: usize,
    values: Vec<L4<T>>,
    geometry: Vec<C<T>>,
    distances: Vec<C<T>>,
    shifts: Vec<[C<T>; 4]>,
    masses: Vec<C<T>>,
    products: Vec<C<T>>,
    poly_a: Vec<L4<T>>,
    poly_b: Vec<L4<T>>,
    poly_h: Vec<L4<T>>,
    poly_u: Vec<L4<T>>,
    poly_v: Vec<L4<T>>,
    matrix: Vec<C<T>>,
    rhs: Vec<C<T>>,
    native_constants: NativeConstants<T>,
    /// Minimal accepted relative determinant raised to its recursion depth.
    stability_threshold: T,
    profiling: bool,
    profile: DirectProfile,
}

struct NativeConstants<T> {
    a: T,
    a_half_squared_minus_b: T,
    normalization: C<T>,
}

impl<T: NativeFloat> DirectExecutor<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(
        plan: Arc<DirectPlanData>,
        prototype: T,
        policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        let zero = czero(&prototype);
        let one = C::new(prototype.one(), prototype.zero());
        let constants = plan
            .constants
            .iter()
            .map(|value| prototype.from_rational(value))
            .collect();
        let families = [
            ScalarIntegral::A0,
            ScalarIntegral::B0,
            ScalarIntegral::C0,
            ScalarIntegral::D0,
        ];
        let mut evaluators: [Option<MasterEvaluator<T>>; 4] = Default::default();
        for (family_index, family) in families.into_iter().enumerate() {
            if !plan.masters[family_index].is_empty() {
                evaluators[family_index] = Some(
                    MasterEvaluator::new_with_policy(family, prototype.clone(), policy)
                        .map_err(|error| ReductionError::Preparation(error.to_string()))?,
                );
            }
        }
        let mut master_offsets = [0; 4];
        let mut master_count = 0;
        for family in 0..4 {
            master_offsets[family] = master_count;
            master_count += plan.masters[family].len();
        }
        let pi = prototype.pi();
        let a = prototype.euler() + &pi.log();
        let b = pi.clone() * &pi / &prototype.from_usize(12);
        let half = prototype.one() / &prototype.from_usize(2);
        let a_half_squared_minus_b = a.clone() * &a * &half - &b;
        let mut pi_power = prototype.one();
        for _ in 0..2 + i32::from(plan.settings.dimension_shift) {
            pi_power *= &pi;
        }
        let normalization = C::new(prototype.zero(), pi_power);
        let gather_degree = plan.basis.len();
        let max_size = plan.subsets.iter().map(|s| s.members.len()).max().unwrap_or(1);
        let products_len = plan.gather_scratch.max(1);
        let stability_threshold = prototype.from_rational(&symbolica::domains::rational::Rational::from((
            symbolica::domains::integer::Integer::from(1),
            symbolica::domains::integer::Integer::from(1_000_000_000_000i64),
        )));
        Ok(Self {
            values: vec![lzero(&zero); plan.value_count],
            geometry: vec![zero.clone(); plan.geometry_count.max(1)],
            distances: vec![zero.clone(); plan.settings.propagators * plan.settings.propagators],
            shifts: vec![[zero.clone(), zero.clone(), zero.clone(), zero.clone()]; plan.settings.propagators],
            masses: vec![zero.clone(); plan.settings.propagators],
            products: vec![zero.clone(); products_len],
            poly_a: vec![lzero(&zero); gather_degree],
            poly_b: vec![lzero(&zero); gather_degree],
            poly_h: vec![lzero(&zero); gather_degree],
            poly_u: vec![lzero(&zero); gather_degree * (usize::from(plan.settings.rank) + 1)],
            poly_v: vec![lzero(&zero); gather_degree * (usize::from(plan.settings.rank) + 1)],
            matrix: vec![zero.clone(); max_size.max(5) * max_size.max(5)],
            rhs: vec![zero.clone(); 5 * 5],
            master_inputs: Default::default(),
            master_values: Vec::new(),
            master_errors: Vec::new(),
            master_offsets,
            master_count,
            native_constants: NativeConstants {
                a,
                a_half_squared_minus_b,
                normalization,
            },
            stability_threshold,
            profiling: false,
            profile: DirectProfile::default(),
            constants,
            evaluators,
            prototype,
            zero,
            one,
            plan,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    pub fn last_profile(&self) -> DirectProfile {
        self.profile
    }

    pub fn set_stability_threshold(&mut self, threshold: T) {
        self.stability_threshold = threshold;
    }

    fn timer(&self) -> Option<Instant> {
        self.profiling.then(Instant::now)
    }

    fn read_lane(&mut self, batch: &KinematicsBatch<T>, row: usize) -> Result<(), String> {
        let n = self.plan.settings.propagators;
        let width = 5 * n + 1;
        let values = &batch.values[row * width..(row + 1) * width];
        for (i, shift) in self.shifts.iter_mut().enumerate() {
            for a in 0..4 {
                shift[a] = values[4 * i + a].clone();
            }
        }
        for i in 0..n {
            self.masses[i] = values[4 * n + i].clone();
        }
        if values.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
            return Err("nonfinite kinematic input".into());
        }
        for i in 0..n {
            for j in 0..n {
                let mut value = self.zero.clone();
                for a in 0..4 {
                    let delta = csub(&self.shifts[i][a], &self.shifts[j][a]);
                    let square = cmul(&delta, &delta);
                    value = if a == 0 {
                        cadd(&value, &square)
                    } else {
                        csub(&value, &square)
                    };
                }
                self.distances[i * n + j] = value;
            }
        }
        Ok(())
    }

    fn mu_squared(&self, batch: &KinematicsBatch<T>, row: usize) -> C<T> {
        let n = self.plan.settings.propagators;
        let mu = &batch.values[row * (5 * n + 1) + 5 * n];
        cmul(mu, mu)
    }

    fn evaluate_masters(&mut self, batch: &KinematicsBatch<T>) -> Result<(), ReductionError> {
        let n = self.plan.settings.propagators;
        let rows = batch.rows;
        for family in 0..4 {
            self.master_inputs[family].clear();
        }
        self.master_values.clear();
        self.master_values
            .resize(rows * self.master_count, lzero(&self.zero));
        self.master_errors.clear();
        self.master_errors.resize(rows, None);
        let plan = self.plan.clone();
        for row in 0..rows {
            if let Err(error) = self.read_lane(batch, row) {
                self.master_errors[row] = Some(error);
            }
            let mu2 = self.mu_squared(batch, row);
            for family in 0..4 {
                for &(subset, _) in &plan.masters[family] {
                    let members = &plan.subsets[subset].members;
                    let d = |i: usize, j: usize| self.distances[members[i] * n + members[j]].clone();
                    let input = &mut self.master_inputs[family];
                    match family {
                        0 => {}
                        1 => input.push(d(0, 1)),
                        2 => input.extend([d(0, 1), d(1, 2), d(0, 2)]),
                        _ => input.extend([d(0, 1), d(1, 2), d(2, 3), d(0, 3), d(0, 2), d(1, 3)]),
                    }
                    input.extend(members.iter().map(|&i| self.masses[i].clone()));
                    input.push(mu2.clone());
                }
            }
        }
        for family in 0..4 {
            let Some(evaluator) = self.evaluators[family].as_mut() else {
                continue;
            };
            let per_lane = plan.masters[family].len();
            let batch_rows = rows * per_lane;
            let result = evaluator
                .evaluate_batch(&self.master_inputs[family], batch_rows)
                .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
            for row in 0..rows {
                for (k, &(subset, _)) in plan.masters[family].iter().enumerate() {
                    let index = row * self.master_count + self.master_offsets[family] + k;
                    match result.lane(row * per_lane + k) {
                        Ok(lane) => {
                            let members = &plan.subsets[subset].members;
                            let mut value = [
                                lane.uv[0].clone(),
                                lane.ir[0].clone(),
                                lane.ir2[0].clone(),
                                lane.finite[0].clone(),
                            ];
                            // Physical pole slots, as in the symbolic planner.
                            match family {
                                0 => {
                                    value[0] = self.master_inputs[0][(row * per_lane + k) * 2].clone();
                                    value[1] = self.zero.clone();
                                    value[2] = self.zero.clone();
                                }
                                1 => value[0] = self.one.clone(),
                                _ => value[0] = self.zero.clone(),
                            }
                            let _ = members;
                            self.master_values[index] = value;
                        }
                        Err(error) => {
                            if self.master_errors[row].is_none() {
                                self.master_errors[row] = Some(format!("{error}"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute every subset's numeric geometry. Returns an instability note.
    fn geometry(&mut self) -> Result<(), String> {
        let plan = self.plan.clone();
        let n = plan.settings.propagators;
        let threshold = self.stability_threshold.clone();
        for subset in &plan.subsets {
            let members = &subset.members;
            let origin = members[0];
            let d = |i: usize, j: usize| &self.distances[i * n + j];
            match subset.kind {
                SubsetKind::Pv => {
                    let geometry = subset.pv.unwrap();
                    let momenta = geometry.n;
                    self.geometry[geometry.m0 as usize] = self.masses[origin].clone();
                    for i in 0..momenta {
                        let member = members[i + 1];
                        self.geometry[geometry.f as usize + i] = cadd(
                            &csub(d(origin, member), &self.masses[member]),
                            &self.masses[origin],
                        );
                        for a in 0..4 {
                            self.geometry[geometry.p as usize + 4 * i + a] =
                                csub(&self.shifts[member][a], &self.shifts[origin][a]);
                        }
                    }
                    for a in 0..4 {
                        self.geometry[geometry.origin as usize + a] = self.shifts[origin][a].clone();
                    }
                    if subset.coefficient_rank >= 1 && momenta > 0 {
                        let size = momenta;
                        for i in 0..size {
                            for j in 0..size {
                                let (mi, mj) = (members[i + 1], members[j + 1]);
                                self.matrix[i * size + j] =
                                    csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
                                self.rhs[i * size + j] =
                                    if i == j { self.one.clone() } else { self.zero.clone() };
                            }
                        }
                        let quality = solve_in_place(
                            &mut self.matrix[..size * size],
                            &mut self.rhs[..size * size],
                            size,
                            size,
                        )
                        .ok_or_else(|| format!("singular Gram matrix for {members:?}"))?;
                        let power = subset.coefficient_rank as u64;
                        if quality.relative_determinant.pow(power) < threshold {
                            return Err(format!(
                                "ill-conditioned Gram matrix for {members:?}: relative determinant {:e}",
                                quality.relative_determinant
                            ));
                        }
                        for j in 0..size {
                            let mut zs = self.zero.clone();
                            let mut zf = self.zero.clone();
                            for i in 0..size {
                                let value = self.rhs[j * size + i].clone();
                                zs = cadd(&zs, &value);
                                cfma(&mut zf, &value, &self.geometry[geometry.f as usize + i]);
                                self.geometry[geometry.zinv as usize + j * size + i] = value;
                            }
                            self.geometry[geometry.zs as usize + j] = zs;
                            self.geometry[geometry.zf as usize + j] = zf;
                        }
                    }
                }
                SubsetKind::Pentagon => {
                    let geometry = subset.pentagon.unwrap();
                    // b = R^-1 1 for the scalar pentagon.
                    for i in 0..5 {
                        for j in 0..5 {
                            let (mi, mj) = (members[i], members[j]);
                            self.matrix[i * 5 + j] = csub(
                                &csub(d(mi, mj), &self.masses[mi]),
                                &self.masses[mj],
                            );
                        }
                        self.rhs[i] = self.one.clone();
                    }
                    let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..5], 5, 1)
                        .ok_or_else(|| format!("singular Cayley matrix for {members:?}"))?;
                    if quality.relative_determinant < threshold {
                        return Err(format!(
                            "ill-conditioned Cayley matrix for {members:?}: relative determinant {:e}",
                            quality.relative_determinant
                        ));
                    }
                    for i in 0..5 {
                        self.geometry[geometry.b as usize + i] = self.rhs[i].clone();
                    }
                    if subset.moment_rank >= 1 {
                        let mut f = [self.zero.clone(), self.zero.clone(), self.zero.clone(), self.zero.clone()];
                        for i in 0..4 {
                            let member = members[i + 1];
                            f[i] = cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
                            for j in 0..4 {
                                let (mi, mj) = (members[i + 1], members[j + 1]);
                                self.matrix[i * 4 + j] =
                                    csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
                                self.rhs[i * 4 + j] =
                                    if i == j { self.one.clone() } else { self.zero.clone() };
                            }
                        }
                        let quality = solve_in_place(&mut self.matrix[..16], &mut self.rhs[..16], 4, 4)
                            .ok_or_else(|| format!("singular pentagon Gram matrix for {members:?}"))?;
                        if quality.relative_determinant.pow(subset.moment_rank as u64) < threshold {
                            return Err(format!(
                                "ill-conditioned pentagon Gram matrix for {members:?}: relative determinant {:e}",
                                quality.relative_determinant
                            ));
                        }
                        for mu in 0..4 {
                            let mut a = csub(&self.zero, &self.shifts[origin][mu]);
                            let mut s = self.zero.clone();
                            for j in 0..4 {
                                let mut w = self.zero.clone();
                                for i in 0..4 {
                                    let p = csub(&self.shifts[members[i + 1]][mu], &self.shifts[origin][mu]);
                                    cfma(&mut w, &p, &self.rhs[i * 4 + j]);
                                }
                                let wf = cmul(&w, &f[j]);
                                a = csub(&a, &wf);
                                s = cadd(&s, &w);
                                self.geometry[geometry.w as usize + 4 * mu + j] = w;
                            }
                            self.geometry[geometry.a as usize + mu] = a;
                            self.geometry[geometry.s as usize + mu] = s;
                        }
                    }
                }
                SubsetKind::Hexagon => {
                    let weights = subset.hexagon.unwrap() as usize;
                    let half = self.prototype.one() / &self.prototype.from_usize(2);
                    for column in 0..5 {
                        let member = members[column + 1];
                        for mu in 0..4 {
                            self.matrix[mu * 5 + column] =
                                csub(&self.shifts[member][mu], &self.shifts[origin][mu]);
                        }
                        self.matrix[4 * 5 + column] =
                            cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
                    }
                    for row in 0..5 {
                        for column in 0..5 {
                            self.rhs[row * 5 + column] = if column == 0 {
                                if row == 4 { self.one.clone() } else { self.zero.clone() }
                            } else if row == 4 {
                                csub(&self.zero, &self.shifts[origin][column - 1])
                            } else if row == column - 1 {
                                let sign = if row == 0 { half.clone() } else { -half.clone() };
                                C::new(sign, self.prototype.zero())
                            } else {
                                self.zero.clone()
                            };
                        }
                    }
                    let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..25], 5, 5)
                        .ok_or_else(|| format!("singular six-point coordinate system for {members:?}"))?;
                    if quality.relative_determinant < threshold {
                        return Err(format!(
                            "ill-conditioned six-point coordinate system: relative determinant {:e}",
                            quality.relative_determinant
                        ));
                    }
                    for column in 0..5 {
                        let mut sum = self.zero.clone();
                        for i in 0..5 {
                            let value = self.rhs[i * 5 + column].clone();
                            sum = cadd(&sum, &value);
                            self.geometry[weights + 6 * column + i + 1] = value;
                        }
                        self.geometry[weights + 6 * column] = csub(&self.zero, &sum);
                    }
                }
            }
        }
        Ok(())
    }

    fn run_program(&mut self, program_is_moments: bool) {
        let plan = self.plan.clone();
        let program = if program_is_moments {
            &plan.moment_program
        } else {
            &plan.coefficient_program
        };
        for instruction in program {
            let mut accumulator = lzero(&self.zero);
            for term in &plan.terms[instruction.terms.0 as usize..instruction.terms.1 as usize] {
                let constant = &self.constants[term.constant as usize];
                let factor = if term.geometry == NO_GEOMETRY {
                    C::new(constant.clone(), self.prototype.zero())
                } else if term.constant == 0 {
                    self.geometry[term.geometry as usize].clone()
                } else {
                    cscale(&self.geometry[term.geometry as usize], constant)
                };
                lfma(&mut accumulator, &factor, &self.values[term.src as usize]);
            }
            match instruction.post {
                PostOp::None => {}
                PostOp::DivideDimension { inverse } => {
                    div_dimension(&mut accumulator, &self.constants[inverse as usize]);
                }
            }
            self.values[instruction.dst as usize] = accumulator;
        }
    }

    fn gather(&mut self) {
        let plan = self.plan.clone();
        let table = &plan.monomials;
        for gather in &plan.gathers {
            let subset = &plan.subsets[gather.subset];
            let geometry = subset.pv.unwrap();
            let rank = usize::from(gather.moment_rank);
            // Momentum products prod_i (p_i.z)^k_i / k_i!.
            for (k, kappa) in gather.kappas.iter().enumerate() {
                let offset = kappa.offset as usize;
                let block = table.block(usize::from(kappa.degree));
                if kappa.degree == 0 {
                    self.products[offset] = self.one.clone();
                    continue;
                }
                for slot in &mut self.products[offset..offset + block.len()] {
                    *slot = self.zero.clone();
                }
                let parent = &gather.kappas[kappa.parent as usize];
                let parent_block = table.block(usize::from(parent.degree));
                let divisor = self.prototype.from_usize(usize::from(kappa.divisor));
                let p = geometry.p as usize + 4 * usize::from(kappa.momentum);
                for (local, monomial) in parent_block.clone().enumerate() {
                    let value = self.products[parent.offset as usize + local].clone();
                    if value.re.is_zero() && value.im.is_zero() {
                        continue;
                    }
                    let value = C::new(value.re / &divisor, value.im / &divisor);
                    for a in 0..4 {
                        let target = table.multiply[monomial][a] as usize - block.start;
                        let update = cmul(&value, &self.geometry[p + a]);
                        self.products[offset + target] =
                            cadd(&self.products[offset + target], &update);
                    }
                }
                let _ = k;
            }
            // Untranslated generating polynomial H_k, Horner in the metric.
            let kappas = gather.kappas.len();
            for degree in 0..=rank {
                let block = table.block(degree);
                for slot in &mut self.poly_h[block.clone()] {
                    *slot = lzero(&self.zero);
                }
                let lambda_max = degree / 2;
                for lambda in (0..=lambda_max).rev() {
                    let momentum_degree = degree - 2 * lambda;
                    let a_block = table.block(momentum_degree);
                    for slot in &mut self.poly_a[a_block.clone()] {
                        *slot = lzero(&self.zero);
                    }
                    for (k, kappa) in gather.kappas.iter().enumerate() {
                        if usize::from(kappa.degree) != momentum_degree {
                            continue;
                        }
                        let coefficient = gather.coefficient[lambda * kappas + k];
                        let value = self.values[coefficient as usize].clone();
                        for (local, monomial) in a_block.clone().enumerate() {
                            let product = &self.products[kappa.offset as usize + local];
                            if product.re.is_zero() && product.im.is_zero() {
                                continue;
                            }
                            lfma(&mut self.poly_a[monomial], product, &value);
                        }
                    }
                    if lambda == lambda_max {
                        for monomial in a_block.clone() {
                            self.poly_b[monomial] = self.poly_a[monomial].clone();
                        }
                    } else {
                        // B <- A + Q B / (2 (lambda + 1)), Q = z0^2 - z^2.
                        let previous = table.block(momentum_degree - 2);
                        let scale = self.prototype.one()
                            / &self.prototype.from_usize(2 * (lambda + 1));
                        for monomial in a_block.clone() {
                            self.poly_h[monomial] = self.poly_a[monomial].clone();
                        }
                        for monomial in previous.clone() {
                            let value = self.poly_b[monomial].clone();
                            for a in 0..4 {
                                let target = table.multiply[table.multiply[monomial][a] as usize][a]
                                    as usize;
                                let factor = if a == 0 { scale.clone() } else { -scale.clone() };
                                for slot in 0..4 {
                                    let update = cscale(&value[slot], &factor);
                                    self.poly_h[target][slot] =
                                        cadd(&self.poly_h[target][slot], &update);
                                }
                            }
                        }
                        for monomial in a_block.clone() {
                            self.poly_b[monomial] = self.poly_h[monomial].clone();
                        }
                    }
                }
                for monomial in block.clone() {
                    self.poly_h[monomial] = self.poly_b[monomial].clone();
                }
            }
            // Translate l -> l - origin: F_r = sum_j T^j/j! H_(r-j), T = -origin.z.
            let origin = geometry.origin as usize;
            let translated = (0..4).any(|a| {
                let value = &self.geometry[origin + a];
                !(value.re.is_zero() && value.im.is_zero())
            });
            let moment_base = gather.moment_base as usize;
            if !translated {
                for monomial in table.block(0).start..table.block(rank).end {
                    let factorial = self.prototype.from_usize(table.factorials[monomial] as usize);
                    let value = &self.poly_h[monomial];
                    self.values[moment_base + monomial] =
                        std::array::from_fn(|slot| cscale(&value[slot], &factorial));
                }
                continue;
            }
            let negative_origin: [C<T>; 4] =
                std::array::from_fn(|a| csub(&self.zero, &self.geometry[origin + a]));
            let stride = table.block(rank).end;
            // poly_u[j * stride + m] holds T^j/j! H_(r-1-j); poly_v the next r.
            for r in 0..=rank {
                let block = table.block(r);
                for j in 0..=r {
                    for monomial in block.clone() {
                        self.poly_v[j * stride + monomial] = lzero(&self.zero);
                    }
                }
                for monomial in block.clone() {
                    self.poly_v[monomial] = self.poly_h[monomial].clone();
                }
                if r > 0 {
                    let previous = table.block(r - 1);
                    for j in 1..=r {
                        let inverse = self.prototype.one() / &self.prototype.from_usize(j);
                        for monomial in previous.clone() {
                            let value = self.poly_u[(j - 1) * stride + monomial].clone();
                            for a in 0..4 {
                                let target = table.multiply[monomial][a] as usize;
                                let factor = cscale(&negative_origin[a], &inverse);
                                lfma(&mut self.poly_v[j * stride + target], &factor, &value);
                            }
                        }
                    }
                }
                for monomial in block.clone() {
                    let mut total = lzero(&self.zero);
                    for j in 0..=r {
                        for slot in 0..4 {
                            total[slot] = cadd(&total[slot], &self.poly_v[j * stride + monomial][slot]);
                        }
                    }
                    let factorial = self.prototype.from_usize(table.factorials[monomial] as usize);
                    self.values[moment_base + monomial] =
                        std::array::from_fn(|slot| cscale(&total[slot], &factorial));
                }
                std::mem::swap(&mut self.poly_u, &mut self.poly_v);
            }
        }
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let n = self.plan.settings.propagators;
        if batch.components != 1 {
            return Err(ReductionError::Unsupported(
                "the direct engine evaluates ordinary numbers; use the generic workspace for Dual components".into(),
            ));
        }
        let width = 5 * n + 1;
        if batch.values.len() != batch.rows * width {
            return Err(ReductionError::InvalidInput(format!(
                "expected {} values per row",
                width
            )));
        }
        self.profile = DirectProfile::default();
        let started = self.timer();
        self.evaluate_masters(batch)?;
        if let Some(started) = started {
            self.profile.masters += started.elapsed();
        }
        let plan = self.plan.clone();
        let basis_len = plan.basis.len();
        let mut values = Vec::with_capacity(batch.rows * basis_len * 4);
        let mut statuses = Vec::with_capacity(batch.rows);
        let settings = &plan.settings;
        let base_uv_degree = 4 + 2 * i32::from(settings.dimension_shift)
            - 2 * settings.powers.iter().map(|&p| i32::from(p)).sum::<i32>();
        for row in 0..batch.rows {
            let lane_values = values.len();
            values.resize(lane_values + basis_len * 4, self.zero.clone());
            if let Some(error) = &self.master_errors[row] {
                statuses.push(LaneStatus::MasterFailure(error.clone()));
                continue;
            }
            let started = self.timer();
            if let Err(error) = self.read_lane(batch, row) {
                statuses.push(LaneStatus::InvalidInput(error));
                continue;
            }
            if let Err(error) = self.geometry() {
                statuses.push(LaneStatus::Unstable(error));
                continue;
            }
            for family in 0..4 {
                for (k, &(_, index)) in plan.masters[family].iter().enumerate() {
                    self.values[index as usize] = self.master_values
                        [row * self.master_count + self.master_offsets[family] + k]
                        .clone();
                }
            }
            if let Some(started) = started {
                self.profile.geometry += started.elapsed();
            }
            let started = self.timer();
            self.run_program(false);
            if let Some(started) = started {
                self.profile.recurrence += started.elapsed();
            }
            let started = self.timer();
            self.gather();
            if let Some(started) = started {
                self.profile.gather += started.elapsed();
            }
            let started = self.timer();
            self.run_program(true);
            if let Some(started) = started {
                self.profile.recurrence += started.elapsed();
            }
            let started = self.timer();
            let mut finite = true;
            for (index, monomial) in plan.basis.iter().enumerate() {
                let mut value = self.values[plan.output[index] as usize].clone();
                let rank = monomial.rank() as i32;
                if base_uv_degree + rank < 0 {
                    value[0] = self.zero.clone();
                }
                match settings.convention {
                    Convention::OneLoop => {}
                    Convention::MadLoop => {
                        if rank % 2 == 1 {
                            for slot in &mut value {
                                *slot = csub(&self.zero, slot);
                            }
                        }
                    }
                    Convention::IregiNative => {
                        let constants = &self.native_constants;
                        let [uv, ir1, ir2, fin] = value.clone();
                        let finite_part = cadd(
                            &csub(&fin, &cscale(&cadd(&uv, &ir1), &constants.a)),
                            &cscale(&ir2, &constants.a_half_squared_minus_b),
                        );
                        let single_ir = csub(&ir1, &cscale(&ir2, &constants.a));
                        value = [uv, single_ir, ir2, finite_part]
                            .map(|slot| cmul(&slot, &constants.normalization));
                    }
                }
                if settings.tensor_weights == TensorWeights::Multinomial {
                    let weight = multinomial(monomial.0);
                    let weight = self.prototype.from_usize(weight as usize);
                    for slot in &mut value {
                        *slot = cscale(slot, &weight);
                    }
                }
                for (slot, entry) in value.into_iter().enumerate() {
                    finite &= entry.re.is_finite() && entry.im.is_finite();
                    values[lane_values + 4 * index + slot] = entry;
                }
            }
            if let Some(started) = started {
                self.profile.output += started.elapsed();
            }
            statuses.push(if finite {
                LaneStatus::Success
            } else {
                LaneStatus::Unstable("nonfinite direct-engine coefficient".into())
            });
        }
        let rows = batch.rows;
        let _ = cnorm2::<T>;
        Ok(BatchResult {
            basis: plan.basis.clone(),
            components: 1,
            metadata: ResultMetadata::new(settings.convention, settings.tensor_weights, None),
            values,
            statuses,
            fallback_used: vec![false; rows],
            analytic_used: vec![false; rows],
            primal_envelopes: vec![None; rows],
            primal_envelope_errors: vec![None; rows],
        })
    }
}

fn multinomial(e: [u16; 4]) -> u64 {
    let factorial = |n: u16| (1..=u64::from(n)).product::<u64>();
    let rank: u16 = e.iter().sum();
    factorial(rank) / e.iter().map(|&x| factorial(x)).product::<u64>()
}
