//! Structural preparation of the direct engine. No sampled value is stored.
//!
//! The plan mirrors the validated symbolic PV planner in `pv.rs`:
//! Denner--Dittmaier Eqs. (5.10)--(5.11) for at most four denominators, the
//! four-dimensional external-basis recursion for pentagon Cartesian moments
//! and the exact thin six-point identity `l^mu = sum_i w_i^mu D_i`. Instead of
//! building one expression per moment, every recurrence node becomes a short
//! linear instruction over a flat value array, evaluated numerically per lane.

use std::collections::HashMap;

use symbolica::domains::{integer::Integer, rational::Rational};

use crate::{
    Algorithm, InputPolicy, InvariantInput, ReductionError, ReductionSettings,
    basis::{Monomial, tensor_basis},
};

/// Largest point count handled by the direct engine.
pub(crate) const MAX_PROPAGATORS: usize = 6;
/// Largest tensor rank handled by the direct engine.
pub(crate) const MAX_RANK: u16 = 8;
pub(crate) const NO_GEOMETRY: u32 = u32::MAX;

/// Dense monomial enumeration in `tensor_basis` order with multiplication maps.
#[derive(Clone, Debug)]
pub(crate) struct MonomialTable {
    pub exponents: Vec<[u16; 4]>,
    /// `degree_start[d]..degree_start[d + 1]` holds the degree-`d` block.
    pub degree_start: Vec<usize>,
    /// Index after multiplying by `z_a`; `u32::MAX` beyond the maximal degree.
    pub multiply: Vec<[u32; 4]>,
    /// `e!` for each monomial.
    pub factorials: Vec<u64>,
}

impl MonomialTable {
    pub fn new(max_degree: u16) -> Self {
        let exponents: Vec<_> = tensor_basis(max_degree).into_iter().map(|m| m.0).collect();
        let lookup: HashMap<[u16; 4], u32> = exponents
            .iter()
            .enumerate()
            .map(|(index, &e)| (e, index as u32))
            .collect();
        let mut degree_start = vec![0; usize::from(max_degree) + 2];
        for (index, e) in exponents.iter().enumerate() {
            let degree = e.iter().map(|&x| usize::from(x)).sum::<usize>();
            degree_start[degree + 1] = index + 1;
        }
        let multiply = exponents
            .iter()
            .map(|e| {
                std::array::from_fn(|a| {
                    let mut next = *e;
                    next[a] += 1;
                    lookup.get(&next).copied().unwrap_or(u32::MAX)
                })
            })
            .collect();
        let factorials = exponents
            .iter()
            .map(|e| e.iter().map(|&x| factorial_u64(x)).product())
            .collect();
        Self {
            exponents,
            degree_start,
            multiply,
            factorials,
        }
    }

    pub fn block(&self, degree: usize) -> std::ops::Range<usize> {
        self.degree_start[degree]..self.degree_start[degree + 1]
    }
}

fn factorial_u64(n: u16) -> u64 {
    (1..=u64::from(n)).product()
}

/// Post-processing of one accumulated instruction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PostOp {
    None,
    /// Multiply by `(a - 2 epsilon)^-1`, with `1/a` stored as a constant.
    DivideDimension { inverse: u32 },
}

/// `value[src] * geometry[geometry] * constant[constant]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Term {
    pub src: u32,
    pub geometry: u32,
    pub constant: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instruction {
    pub dst: u32,
    pub terms: (u32, u32),
    pub post: PostOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubsetKind {
    /// At most four denominators: PV coefficients and a Cartesian gather.
    Pv,
    Pentagon,
    Hexagon,
    /// Affinely dependent external points: the lane's constant-one
    /// denominator identity `sum_i w_i D_i = 1` reduces every moment to
    /// same-rank pinches. Index into `DirectPlanData::relations`.
    Relation(usize),
}

/// A minimal affinely dependent propagator subset of a plan variant.
#[derive(Clone, Debug)]
pub(crate) struct RelationGeometry {
    pub mask: u8,
    pub members: Vec<usize>,
    /// Geometry offset of the identity weights, one per member.
    pub weights: u32,
}

/// Geometry offsets of a PV subset with `n` momenta.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PvGeometry {
    pub n: usize,
    pub m0: u32,
    pub f: u32,
    pub zinv: u32,
    pub zs: u32,
    pub zf: u32,
    pub p: u32,
    pub origin: u32,
}

impl PvGeometry {
    fn new(base: u32, n: usize) -> (Self, u32) {
        let n32 = n as u32;
        let m0 = base;
        let f = m0 + 1;
        let zinv = f + n32;
        let zs = zinv + n32 * n32;
        let zf = zs + n32;
        let p = zf + n32;
        let origin = p + 4 * n32;
        (
            Self {
                n,
                m0,
                f,
                zinv,
                zs,
                zf,
                p,
                origin,
            },
            origin + 4 - base,
        )
    }
}

/// Geometry offsets of a pentagon: `W[mu][j]`, `a[mu]`, `s[mu]`, `b[i]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PentagonGeometry {
    pub w: u32,
    pub a: u32,
    pub s: u32,
    pub b: u32,
}

/// One gather of Cartesian moments from PV coefficients.
#[derive(Clone, Debug)]
pub(crate) struct Gather {
    pub subset: usize,
    pub moment_rank: u16,
    /// Momentum multi-indices by degree, then lexicographically.
    pub kappas: Vec<GatherKappa>,
    /// Coefficient value index for `(lambda, kappa)`; `u32::MAX` if unused.
    /// Row-major `[lambda][kappa]`, `lambda <= moment_rank / 2`.
    pub coefficient: Vec<u32>,
    pub moment_base: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherKappa {
    pub degree: u16,
    /// Parent multi-index and the momentum multiplied in; none for zero.
    pub parent: u32,
    pub momentum: u16,
    /// Exponent of `momentum` in this multi-index (the `1/kappa!` divisor).
    pub divisor: u16,
    /// Offset of this product polynomial's degree block in the scratch buffer.
    pub offset: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct Subset {
    pub mask: u8,
    pub members: Vec<usize>,
    pub kind: SubsetKind,
    pub coefficient_rank: i32,
    pub moment_rank: i32,
    pub geometry_base: u32,
    pub pv: Option<PvGeometry>,
    pub pentagon: Option<PentagonGeometry>,
    /// Hexagon thin weights `w[column][i]`, 5 columns x 6 members.
    pub hexagon: Option<u32>,
    /// Value index receiving the scalar master, for PV subsets.
    pub master: Option<u32>,
    /// Value index of the first moment, in `tensor_basis` order.
    pub moment_base: Option<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct DirectPlanData {
    pub settings: ReductionSettings,
    pub basis: Vec<Monomial>,
    pub monomials: MonomialTable,
    pub subsets: Vec<Subset>,
    pub value_count: usize,
    pub geometry_count: usize,
    pub constants: Vec<Rational>,
    /// Coefficient recursions, executed before gathers.
    pub coefficient_program: Vec<Instruction>,
    pub gathers: Vec<Gather>,
    /// Pentagon and hexagon moment recursions, executed after gathers.
    pub moment_program: Vec<Instruction>,
    pub terms: Vec<Term>,
    /// Masters per family `A0..D0`: `(subset, value index)`.
    pub masters: [Vec<(usize, u32)>; 4],
    /// Minimal degenerate subsets of this variant, in profile order.
    pub relations: Vec<RelationGeometry>,
    pub output: Vec<u32>,
    pub gather_scratch: usize,
}

struct Builder {
    constants: Vec<Rational>,
    constant_lookup: HashMap<Rational, u32>,
    terms: Vec<Term>,
}

impl Builder {
    fn constant(&mut self, value: Rational) -> u32 {
        if let Some(&index) = self.constant_lookup.get(&value) {
            return index;
        }
        let index = self.constants.len() as u32;
        self.constants.push(value.clone());
        self.constant_lookup.insert(value, index);
        index
    }

    fn term(&mut self, src: u32, geometry: u32, constant: Rational) -> Term {
        let constant = self.constant(constant);
        Term {
            src,
            geometry,
            constant,
        }
    }

    fn instruction(&mut self, dst: u32, terms: Vec<Term>, post: PostOp) -> Instruction {
        let start = self.terms.len() as u32;
        self.terms.extend(terms);
        Instruction {
            dst,
            terms: (start, self.terms.len() as u32),
            post,
        }
    }
}

fn rational(numerator: i64, denominator: i64) -> Rational {
    Rational::from((Integer::from(numerator), Integer::from(denominator)))
}

fn factorial(n: u16) -> Integer {
    Integer::factorial(u32::from(n))
}

fn binomial(n: u16, k: u16) -> Integer {
    factorial(n) / factorial(k) / factorial(n - k)
}

/// Weak compositions of `total` into `count` parts, in `pv.rs` order.
pub(crate) fn compositions(total: u16, count: usize) -> Vec<Vec<u16>> {
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

fn without(mask: u8, member: usize) -> u8 {
    mask & !(1u8 << member)
}

fn members(mask: u8) -> Vec<usize> {
    (0..8).filter(|&i| mask & (1 << i) != 0).collect()
}

/// Per-PV-subset coefficient layout.
struct CoefficientLayout {
    lookup: HashMap<(u16, Vec<u16>), u32>,
    spz: HashMap<(u16, Vec<u16>), u32>,
}

impl DirectPlanData {
    pub fn supported(settings: &ReductionSettings) -> Result<(), ReductionError> {
        let n = settings.propagators;
        if n == 0 || n > MAX_PROPAGATORS {
            return Err(ReductionError::Unsupported(format!(
                "direct engine handles 1..={MAX_PROPAGATORS} propagators"
            )));
        }
        if settings.rank > MAX_RANK {
            return Err(ReductionError::Unsupported(format!(
                "direct engine handles rank <= {MAX_RANK}"
            )));
        }
        if settings.powers.len() != n || settings.powers.iter().any(|&p| p != 1) {
            return Err(ReductionError::Unsupported(
                "direct engine currently requires unit propagator powers".into(),
            ));
        }
        if settings.dimension_shift != 0 {
            return Err(ReductionError::Unsupported(
                "direct engine currently requires dimension shift zero".into(),
            ));
        }
        if settings.algorithm == Algorithm::Ibp {
            return Err(ReductionError::Unsupported(
                "direct engine implements the PV and Hybrid algorithms".into(),
            ));
        }
        if settings.invariants != InvariantInput::FromShifts {
            return Err(ReductionError::Unsupported(
                "direct engine derives invariants from Cartesian shifts".into(),
            ));
        }
        if settings.input_policy != InputPolicy::Preserve {
            return Err(ReductionError::Unsupported(
                "direct engine preserves inputs; legacy snapping uses the symbolic planner".into(),
            ));
        }
        Ok(())
    }

    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::with_relations(settings, &[])
    }

    /// Plan variant in which every subset (of at most five propagators)
    /// containing one of `relation_masks` is reduced by that subset's
    /// constant-one denominator identity instead of its singular Gram system.
    pub fn with_relations(
        settings: ReductionSettings,
        relation_masks: &[u8],
    ) -> Result<Self, ReductionError> {
        Self::supported(&settings)?;
        let relation_for = |mask: u8| -> Option<usize> {
            if mask.count_ones() >= 6 {
                return None;
            }
            relation_masks.iter().position(|&r| r & !mask == 0)
        };
        let n = settings.propagators;
        let rank = settings.rank;
        let basis = tensor_basis(rank);
        let monomials = MonomialTable::new(rank);
        let top: u8 = ((1u16 << n) - 1) as u8;

        // Propagate rank needs from the top topology to its pinches.
        let mut coefficient_need: HashMap<u8, i32> = HashMap::new();
        let mut moment_need: HashMap<u8, i32> = HashMap::new();
        moment_need.insert(top, i32::from(rank));
        let mut masks: Vec<u8> = (1..=top).filter(|m| m & !top == 0).collect();
        masks.sort_by_key(|m| std::cmp::Reverse(m.count_ones()));
        for &mask in &masks {
            let size = mask.count_ones() as usize;
            let moment = moment_need.get(&mask).copied().unwrap_or(-1);
            if let Some(relation) = relation_for(mask) {
                if moment >= 0 {
                    for member in members(relation_masks[relation]) {
                        let child = without(mask, member);
                        let need = moment_need.entry(child).or_insert(-1);
                        *need = (*need).max(moment);
                    }
                }
                continue;
            }
            match size {
                6 | 5 => {
                    if moment >= 0 {
                        for member in members(mask) {
                            let child = without(mask, member);
                            let need = moment_need.entry(child).or_insert(-1);
                            *need = (*need).max((moment - 1).max(0));
                        }
                    }
                }
                _ => {
                    if moment >= 0 {
                        let need = coefficient_need.entry(mask).or_insert(-1);
                        *need = (*need).max(moment);
                    }
                    let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
                    if coefficient >= 1 && size >= 2 {
                        for member in members(mask) {
                            let child = without(mask, member);
                            let need = coefficient_need.entry(child).or_insert(-1);
                            *need = (*need).max(coefficient - 1);
                        }
                    }
                }
            }
        }

        let mut builder = Builder {
            constants: Vec::new(),
            constant_lookup: HashMap::new(),
            terms: Vec::new(),
        };
        builder.constant(rational(1, 1));

        // Subsets in increasing size, so children precede parents.
        masks.reverse();
        let mut subsets = Vec::new();
        let mut subset_index: HashMap<u8, usize> = HashMap::new();
        let mut value_count: u32 = 0;
        let mut geometry_count: u32 = 0;
        let mut layouts: HashMap<u8, CoefficientLayout> = HashMap::new();
        let mut relations = Vec::new();
        for &relation in relation_masks {
            let member_list = members(relation);
            relations.push(RelationGeometry {
                mask: relation,
                weights: geometry_count,
                members: member_list.clone(),
            });
            geometry_count += member_list.len() as u32;
        }
        for &mask in &masks {
            let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
            let moment = moment_need.get(&mask).copied().unwrap_or(-1);
            if coefficient < 0 && moment < 0 {
                continue;
            }
            let member_list = members(mask);
            let size = member_list.len();
            let kind = match (relation_for(mask), size) {
                (Some(relation), _) => SubsetKind::Relation(relation),
                (None, 6) => SubsetKind::Hexagon,
                (None, 5) => SubsetKind::Pentagon,
                _ => SubsetKind::Pv,
            };
            let geometry_base = geometry_count;
            let mut subset = Subset {
                mask,
                members: member_list,
                kind,
                coefficient_rank: coefficient,
                moment_rank: moment,
                geometry_base,
                pv: None,
                pentagon: None,
                hexagon: None,
                master: None,
                moment_base: None,
            };
            match kind {
                SubsetKind::Pv => {
                    let (geometry, length) = PvGeometry::new(geometry_base, size - 1);
                    subset.pv = Some(geometry);
                    geometry_count += length;
                    let momenta = size - 1;
                    let mut layout = CoefficientLayout {
                        lookup: HashMap::new(),
                        spz: HashMap::new(),
                    };
                    for r in 0..=coefficient.max(0) as u16 {
                        for lambda in (0..=r / 2).rev() {
                            for kappa in compositions(r - 2 * lambda, momenta) {
                                layout.lookup.insert((lambda, kappa), value_count);
                                value_count += 1;
                            }
                        }
                    }
                    if size >= 2 && coefficient >= 1 {
                        for r in 0..coefficient as u16 {
                            for lambda in (0..=r / 2).rev() {
                                for kappa in compositions(r - 2 * lambda, momenta) {
                                    layout.spz.insert((lambda, kappa), value_count);
                                    value_count += 1;
                                }
                            }
                        }
                    }
                    subset.master = Some(layout.lookup[&(0, vec![0; momenta])]);
                    layouts.insert(mask, layout);
                }
                SubsetKind::Pentagon => {
                    subset.pentagon = Some(PentagonGeometry {
                        w: geometry_base,
                        a: geometry_base + 16,
                        s: geometry_base + 20,
                        b: geometry_base + 24,
                    });
                    geometry_count += 29;
                }
                SubsetKind::Hexagon => {
                    subset.hexagon = Some(geometry_base);
                    geometry_count += 30;
                }
                SubsetKind::Relation(_) => {
                    if coefficient >= 0 {
                        return Err(ReductionError::Preparation(
                            "a degenerate subset cannot supply PV coefficients".into(),
                        ));
                    }
                }
            }
            if moment >= 0 {
                subset.moment_base = Some(value_count);
                value_count += tensor_basis(moment as u16).len() as u32;
            }
            subset_index.insert(mask, subsets.len());
            subsets.push(subset);
        }

        // Coefficient programs: increasing subset size, then rank.
        let mut coefficient_program = Vec::new();
        let mut masters: [Vec<(usize, u32)>; 4] = Default::default();
        for (id, subset) in subsets.iter().enumerate() {
            if subset.kind != SubsetKind::Pv {
                continue;
            }
            let size = subset.members.len();
            masters[size - 1].push((id, subset.master.unwrap()));
            let geometry = subset.pv.unwrap();
            let layout = &layouts[&subset.mask];
            let momenta = size - 1;
            let c = subset.coefficient_rank.max(0) as u16;
            if size == 1 {
                for lambda in 1..=c / 2 {
                    let dst = layout.lookup[&(lambda, vec![])];
                    let src = layout.lookup[&(lambda - 1, vec![])];
                    let term = builder.term(src, geometry.m0, rational(1, 1));
                    let inverse = builder.constant(rational(1, 2 * i64::from(lambda) + 2));
                    coefficient_program.push(builder.instruction(
                        dst,
                        vec![term],
                        PostOp::DivideDimension { inverse },
                    ));
                }
                continue;
            }
            for r in 1..=c {
                let zero_child = &layouts[&without(subset.mask, subset.members[0])];
                // Shifted pinch-zero values of rank r-1 (Denner Eq. 2.9).
                for lambda in (0..=(r - 1) / 2).rev() {
                    for kappa in compositions(r - 1 - 2 * lambda, momenta) {
                        let dst = layout.spz[&(lambda, kappa.clone())];
                        let first = kappa[0];
                        let sign: i64 = if first % 2 == 0 { 1 } else { -1 };
                        let mut terms = Vec::new();
                        for total in 0..=first {
                            for extra in compositions(total, momenta - 1) {
                                let child: Vec<u16> = kappa[1..]
                                    .iter()
                                    .zip(&extra)
                                    .map(|(&a, &b)| a + b)
                                    .collect();
                                let src = zero_child.lookup[&(lambda, child)];
                                let count = binomial(first, total) * factorial(total)
                                    / extra.iter().fold(Integer::from(1), |acc, &x| acc * factorial(x));
                                let value = Rational::from(count) * &rational(sign, 1);
                                terms.push(builder.term(src, NO_GEOMETRY, value));
                            }
                        }
                        coefficient_program.push(builder.instruction(dst, terms, PostOp::None));
                    }
                }
                // Metric coefficients first, Eq. (5.10).
                for lambda in (1..=r / 2).rev() {
                    for kappa in compositions(r - 2 * lambda, momenta) {
                        let dst = layout.lookup[&(lambda, kappa.clone())];
                        let half = rational(1, 2);
                        let mut terms = vec![builder.term(
                            layout.spz[&(lambda - 1, kappa.clone())],
                            NO_GEOMETRY,
                            half.clone(),
                        )];
                        terms.push(builder.term(
                            layout.lookup[&(lambda - 1, kappa.clone())],
                            geometry.m0,
                            rational(1, 1),
                        ));
                        for i in 0..momenta {
                            let mut child = kappa.clone();
                            child[i] += 1;
                            terms.push(builder.term(
                                layout.lookup[&(lambda - 1, child)],
                                geometry.f + i as u32,
                                half.clone(),
                            ));
                        }
                        let divisor = i64::from(r) + 3 - size as i64;
                        if divisor == 0 {
                            return Err(ReductionError::Preparation(
                                "PV trace recurrence needs a nonzero constant divisor".into(),
                            ));
                        }
                        let inverse = builder.constant(rational(1, divisor));
                        coefficient_program.push(builder.instruction(
                            dst,
                            terms,
                            PostOp::DivideDimension { inverse },
                        ));
                    }
                }
                // Momentum coefficients, Eq. (5.11).
                for kappa in compositions(r, momenta) {
                    let dst = layout.lookup[&(0, kappa.clone())];
                    let j = kappa.iter().position(|&x| x != 0).unwrap();
                    let mut rest = kappa.clone();
                    rest[j] -= 1;
                    let n32 = momenta as u32;
                    let mut terms = vec![
                        builder.term(
                            layout.spz[&(0, rest.clone())],
                            geometry.zs + j as u32,
                            rational(-1, 1),
                        ),
                        builder.term(
                            layout.lookup[&(0, rest.clone())],
                            geometry.zf + j as u32,
                            rational(-1, 1),
                        ),
                    ];
                    for i in 0..momenta {
                        let zinv = geometry.zinv + j as u32 * n32 + i as u32;
                        if rest[i] == 0 {
                            let child_mask = without(subset.mask, subset.members[i + 1]);
                            let mut child = rest.clone();
                            child.remove(i);
                            let src = layouts[&child_mask].lookup[&(0, child)];
                            terms.push(builder.term(src, zinv, rational(1, 1)));
                        } else {
                            let mut child = rest.clone();
                            child[i] -= 1;
                            let src = layout.lookup[&(1, child)];
                            terms.push(builder.term(src, zinv, rational(-2 * i64::from(rest[i]), 1)));
                        }
                    }
                    coefficient_program.push(builder.instruction(dst, terms, PostOp::None));
                }
            }
        }

        // Gathers of PV subsets whose moments are needed.
        let mut gathers = Vec::new();
        let mut gather_scratch = 0usize;
        for (id, subset) in subsets.iter().enumerate() {
            if subset.kind != SubsetKind::Pv || subset.moment_rank < 0 {
                continue;
            }
            let momenta = subset.members.len() - 1;
            let rank = subset.moment_rank as u16;
            let layout = &layouts[&subset.mask];
            let mut kappas = Vec::new();
            let mut kappa_lookup: HashMap<Vec<u16>, u32> = HashMap::new();
            let mut offset = 0u32;
            for degree in 0..=rank {
                for kappa in compositions(degree, momenta) {
                    let (parent, momentum, divisor) = if degree == 0 {
                        (u32::MAX, 0, 1)
                    } else {
                        let last = kappa.iter().rposition(|&x| x != 0).unwrap();
                        let mut parent = kappa.clone();
                        parent[last] -= 1;
                        (kappa_lookup[&parent], last as u16, kappa[last])
                    };
                    kappa_lookup.insert(kappa.clone(), kappas.len() as u32);
                    kappas.push(GatherKappa {
                        degree,
                        parent,
                        momentum,
                        divisor,
                        offset,
                    });
                    offset += monomials.block(usize::from(degree)).len() as u32;
                }
            }
            gather_scratch = gather_scratch.max(offset as usize);
            let lambdas = usize::from(rank / 2) + 1;
            let mut coefficient = vec![u32::MAX; lambdas * kappas.len()];
            for lambda in 0..lambdas as u16 {
                for (kappa, &k) in kappa_lookup.iter() {
                    if 2 * lambda + kappas[k as usize].degree <= rank {
                        coefficient[usize::from(lambda) * kappas.len() + k as usize] =
                            layout.lookup[&(lambda, kappa.clone())];
                    }
                }
            }
            gathers.push(Gather {
                subset: id,
                moment_rank: rank,
                kappas,
                coefficient,
                moment_base: subset.moment_base.unwrap(),
            });
        }

        // Pentagon and hexagon Cartesian moment recursions.
        let mut moment_program = Vec::new();
        let child_moment = |mask: u8, exponents: [u16; 4]| -> u32 {
            let child = &subsets[subset_index[&mask]];
            let rank: u16 = exponents.iter().sum();
            debug_assert!(i32::from(rank) <= child.moment_rank);
            child.moment_base.unwrap() + index_in_basis(exponents)
        };
        for subset in &subsets {
            if subset.moment_rank < 0 {
                continue;
            }
            match subset.kind {
                SubsetKind::Pv => {}
                SubsetKind::Relation(relation) => {
                    let relation = &relations[relation];
                    let base = subset.moment_base.unwrap();
                    for (index, e) in tensor_basis(subset.moment_rank as u16).iter().enumerate() {
                        let terms = relation
                            .members
                            .iter()
                            .enumerate()
                            .map(|(j, &member)| {
                                builder.term(
                                    child_moment(without(subset.mask, member), e.0),
                                    relation.weights + j as u32,
                                    rational(1, 1),
                                )
                            })
                            .collect();
                        moment_program.push(builder.instruction(
                            base + index as u32,
                            terms,
                            PostOp::None,
                        ));
                    }
                }
                SubsetKind::Pentagon => {
                    let geometry = subset.pentagon.unwrap();
                    let base = subset.moment_base.unwrap();
                    for (index, e) in tensor_basis(subset.moment_rank as u16).iter().enumerate() {
                        let dst = base + index as u32;
                        let mut terms = Vec::new();
                        if e.rank() == 0 {
                            for (i, &member) in subset.members.iter().enumerate() {
                                let src = child_moment(without(subset.mask, member), e.0);
                                terms.push(builder.term(src, geometry.b + i as u32, rational(1, 1)));
                            }
                        } else {
                            let mu = e.0.iter().position(|&x| x != 0).unwrap();
                            let mut rest = e.0;
                            rest[mu] -= 1;
                            terms.push(builder.term(
                                base + index_in_basis(rest),
                                geometry.a + mu as u32,
                                rational(1, 1),
                            ));
                            terms.push(builder.term(
                                child_moment(without(subset.mask, subset.members[0]), rest),
                                geometry.s + mu as u32,
                                rational(-1, 1),
                            ));
                            for j in 0..4 {
                                terms.push(builder.term(
                                    child_moment(without(subset.mask, subset.members[j + 1]), rest),
                                    geometry.w + (4 * mu + j) as u32,
                                    rational(1, 1),
                                ));
                            }
                        }
                        moment_program.push(builder.instruction(dst, terms, PostOp::None));
                    }
                }
                SubsetKind::Hexagon => {
                    let weights = subset.hexagon.unwrap();
                    let base = subset.moment_base.unwrap();
                    for (index, e) in tensor_basis(subset.moment_rank as u16).iter().enumerate() {
                        let (column, rest) = match e.0.iter().position(|&x| x != 0) {
                            Some(mu) => {
                                let mut rest = e.0;
                                rest[mu] -= 1;
                                (mu + 1, rest)
                            }
                            None => (0, e.0),
                        };
                        let terms = subset
                            .members
                            .iter()
                            .enumerate()
                            .map(|(i, &member)| {
                                builder.term(
                                    child_moment(without(subset.mask, member), rest),
                                    weights + (6 * column + i) as u32,
                                    rational(1, 1),
                                )
                            })
                            .collect();
                        moment_program.push(builder.instruction(
                            base + index as u32,
                            terms,
                            PostOp::None,
                        ));
                    }
                }
            }
        }

        let top_subset = &subsets[subset_index[&top]];
        let output = (0..basis.len() as u32)
            .map(|i| top_subset.moment_base.unwrap() + i)
            .collect();
        Ok(Self {
            settings,
            basis,
            monomials,
            subsets,
            value_count: value_count as usize,
            geometry_count: geometry_count as usize,
            constants: builder.constants,
            coefficient_program,
            gathers,
            moment_program,
            terms: builder.terms,
            masters,
            relations,
            output,
            gather_scratch,
        })
    }
}

/// Position of an exponent vector in `tensor_basis` (independent of max rank).
pub(crate) fn index_in_basis(e: [u16; 4]) -> u32 {
    let rank: u16 = e.iter().sum();
    let mut offset = 0u32;
    for r in 0..u32::from(rank) {
        offset += (r + 1) * (r + 2) * (r + 3) / 6;
    }
    // Within a rank block: t descending, then x descending, then y descending.
    let rank = u32::from(rank);
    let (t, x, y) = (u32::from(e[0]), u32::from(e[1]), u32::from(e[2]));
    let mut position = 0u32;
    for tt in (t + 1..=rank).rev() {
        let rest = rank - tt;
        position += (rest + 1) * (rest + 2) / 2;
    }
    let rest = rank - t;
    for xx in (x + 1..=rest).rev() {
        position += rest - xx + 1;
    }
    position += rest - x - y;
    offset + position
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_index_matches_enumeration() {
        for (index, monomial) in tensor_basis(7).iter().enumerate() {
            assert_eq!(index_in_basis(monomial.0) as usize, index, "{monomial:?}");
        }
    }

    #[test]
    fn plans_cover_one_to_six_points() {
        for n in 1..=6 {
            for rank in 0..=6 {
                let plan = DirectPlanData::new(ReductionSettings::new(n, rank)).unwrap();
                assert_eq!(plan.output.len(), plan.basis.len());
                assert!(plan.output.iter().all(|&i| (i as usize) < plan.value_count));
            }
        }
    }
}
