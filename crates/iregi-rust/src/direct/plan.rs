//! Structural preparation of the direct engine. No sampled value is stored.
//!
//! The plan mirrors the validated symbolic PV planner in `pv.rs`:
//! Denner--Dittmaier Eqs. (5.10)--(5.11) for at most four denominators, the
//! Gram-free Denner--Dittmaier pentagon reduction (hep-ph/0509141 sect. 5:
//! coefficients up to rank 5 from box coefficients with only the modified
//! Cayley determinant inverted; the four-dimensional external-basis moment
//! recursion continues above rank 5 and for dotted propagators) and the exact
//! thin six-point identity `l^mu = sum_i w_i^mu D_i`. Instead of
//! building one expression per moment, every recurrence node becomes a short
//! linear instruction over a flat value array, evaluated numerically per lane.

use std::collections::{HashMap, HashSet};

use symbolica::domains::{integer::Integer, rational::Rational};

use crate::{
    Algorithm, InputPolicy, InvariantInput, ReductionError, ReductionSettings,
    basis::{Monomial, tensor_basis},
};

/// Fx-style hasher for small integer keys: the coefficient lookups of the
/// small-Gram expansions were dominated by the default SipHash.
#[derive(Clone, Copy, Default)]
pub(crate) struct FastHasher(u64);

impl std::hash::Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.add(u64::from(byte));
        }
    }
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }
    fn write_u16(&mut self, value: u16) {
        self.add(u64::from(value));
    }
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

impl FastHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;

/// A subset of a plan's propagators, bit `i` for propagator `i`. Its width
/// bounds the number of propagators (`SubsetMask::BITS`, reported by
/// [`super::DirectCapabilities`]); the 2^N subsets of a plan bind memory far
/// below it.
pub(crate) type SubsetMask = u64;

/// Every propagator of an `n`-point plan.
pub(crate) fn full_mask(n: usize) -> SubsetMask {
    if n >= SubsetMask::BITS as usize { SubsetMask::MAX } else { ((1 as SubsetMask) << n) - 1 }
}
/// What the direct engine's representation serves: every bound derived from
/// a representation (or an algorithm), none a fixed budget. Rank and point
/// count are otherwise limited by memory (C(R+4, 4) monomials per moment
/// array, 2^N subsets per plan: a plan whose subsets cannot be allocated is
/// refused typed) and by the achieved precision, which callers measure; the
/// validated domain (N <= 6, R <= 6, the opening benchmark) is
/// documentation, not a limit.
///
/// The structural bounds are the same for every carrier. What a carrier
/// adds is its precision and its exponent range: a lane whose coefficients
/// leave the carrier's range (binary64 at very large ranks or dimension
/// shifts, around h = 700) is refused per lane with a typed
/// [`crate::LaneStatus::Unstable`] status ("nonfinite direct-engine
/// coefficient"), never returned as a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCapabilities {
    /// Working precision of the carrier (bits).
    pub carrier_bits: u32,
    /// Largest number of propagators: the width of the subset masks (64).
    pub max_propagators: usize,
    /// Largest tensor rank: monomial and value indices are `u32`, so
    /// C(R + 4, 4) <= u32::MAX.
    pub max_rank: u16,
    /// Propagators that may carry a raised power (one Taylor variable).
    pub max_dotted_propagators: usize,
    /// Largest number of propagators with a raised power (the seven-point
    /// thin system has no mass-derivative jets).
    pub max_dotted_points: usize,
    /// Largest PV coefficient rank `R + 2h` (a rank-`R` integral in
    /// `D + 2h` dimensions needs its subsets' coefficients up to `R + 2h`):
    /// the `u16` coefficient ranks of the plan. It bounds the dimension shift
    /// jointly with the rank, see [`Self::max_dimension_shift`].
    pub max_coefficient_rank: u16,
    /// Largest number of propagators with a dimension shift (a shifted
    /// pentagon is a genuine master, not reducible to boxes).
    pub max_shifted_points: usize,
}

impl DirectCapabilities {
    /// The bounds of the direct engine's representation for a carrier of
    /// `carrier_bits` bits.
    pub fn of_representation(carrier_bits: u32) -> Self {
        // Largest R with C(R + 4, 4) <= u32::MAX.
        let monomials = |r: u64| (r + 1) * (r + 2) * (r + 3) * (r + 4) / 24;
        let mut max_rank = 0u64;
        while max_rank < u64::from(u16::MAX) && monomials(max_rank + 1) <= u64::from(u32::MAX) {
            max_rank += 1;
        }
        Self {
            carrier_bits,
            max_propagators: SubsetMask::BITS as usize,
            max_rank: max_rank as u16,
            max_dotted_propagators: 1,
            max_dotted_points: 6,
            max_coefficient_rank: u16::MAX,
            max_shifted_points: 4,
        }
    }

    /// Largest raised power at `n` propagators: the packed IBP keys give each
    /// propagator 64 / n bits (and the settings hold a `u16`).
    pub fn max_power(&self, n: usize) -> u16 {
        crate::direct::exec_ibp_field_max(n).min(u64::from(u16::MAX)) as u16
    }

    /// Largest dimension shift `h` at tensor rank `rank`: `R + 2h` within
    /// [`Self::max_coefficient_rank`] (and `h` within the settings' `i16`).
    pub fn max_dimension_shift(&self, rank: u16) -> i16 {
        (self.max_coefficient_rank.saturating_sub(rank) / 2).min(i16::MAX as u16) as i16
    }

    /// Whether `settings` lie inside these bounds (the direct engine's
    /// domain): `Ok` exactly when a plan can be built, up to memory.
    pub fn admits(&self, settings: &ReductionSettings) -> Result<(), ReductionError> {
        let n = settings.propagators;
        if n == 0 {
            return Err(ReductionError::Unsupported("direct engine needs at least one propagator".into()));
        }
        if n > self.max_propagators {
            return Err(ReductionError::Unsupported(format!(
                "direct engine: {n} propagators exceed its {}-bit subset masks",
                self.max_propagators
            )));
        }
        if settings.rank > self.max_rank {
            return Err(ReductionError::Unsupported(format!(
                "direct engine: rank {} exceeds the {} monomials its u32 indices address (largest rank {})",
                settings.rank,
                u32::MAX,
                self.max_rank
            )));
        }
        if settings.powers.len() != n || settings.powers.contains(&0) {
            return Err(ReductionError::Unsupported(
                "direct engine requires one positive power per propagator".into(),
            ));
        }
        let dotted = settings.powers.iter().filter(|&&p| p > 1).count();
        if dotted > self.max_dotted_propagators {
            return Err(ReductionError::Unsupported(
                "direct engine handles at most one raised propagator power (one Taylor variable)".into(),
            ));
        }
        if dotted > 0 && n > self.max_dotted_points {
            // The seven-point thin system picks its six members per lane;
            // its mass-derivative jets are not implemented.
            return Err(ReductionError::Unsupported(format!(
                "direct engine handles raised powers up to {} propagators",
                self.max_dotted_points
            )));
        }
        let largest = self.max_power(n);
        if settings.powers.iter().any(|&p| p > largest) {
            return Err(ReductionError::Unsupported(format!(
                "direct engine: a raised power above {largest} exceeds its {}-bit IBP key fields at {n} propagators",
                crate::direct::exec_ibp_field_bits(n)
            )));
        }
        if settings.dimension_shift < 0 {
            return Err(ReductionError::Unsupported(
                "direct engine handles nonnegative dimension shifts".into(),
            ));
        }
        if settings.dimension_shift > self.max_dimension_shift(settings.rank) {
            return Err(ReductionError::Unsupported(format!(
                "direct engine: rank {} in D + 2*{} dimensions needs coefficient rank {} beyond its u16 coefficient ranks (largest shift at this rank {})",
                settings.rank,
                settings.dimension_shift,
                u32::from(settings.rank) + 2 * settings.dimension_shift as u32,
                self.max_dimension_shift(settings.rank)
            )));
        }
        if settings.dimension_shift > 0 && n > self.max_shifted_points {
            // A shifted scalar pentagon is a genuine master, not reducible
            // to four-dimensional boxes.
            return Err(ReductionError::Unsupported(format!(
                "direct engine handles dimension shifts up to {} propagators",
                self.max_shifted_points
            )));
        }
        if settings.algorithm == Algorithm::Ibp {
            return Err(ReductionError::Unsupported(
                "direct engine implements the PV and Hybrid algorithms".into(),
            ));
        }
        if settings.invariants == InvariantInput::SuppliedScalarProducts {
            return Err(ReductionError::Unsupported(
                "direct engine takes invariants from Cartesian shifts or as supplied invariants (SuppliedInvariants), not as supplied scalar products".into(),
            ));
        }
        if settings.input_policy != InputPolicy::Preserve {
            return Err(ReductionError::Unsupported(
                "direct engine preserves inputs; legacy snapping uses the symbolic planner".into(),
            ));
        }
        Ok(())
    }
}
pub(crate) const NO_GEOMETRY: u32 = u32::MAX;
/// Highest pentagon coefficient rank from the Gram-free Denner--Dittmaier
/// reduction (hep-ph/0509141, sect. 5, with the rank-5 UV remainder); higher
/// pentagon ranks continue with the four-dimensional Gram recursion.
pub(crate) const DD_PENTAGON_RANK: i32 = 5;

/// Dense monomial enumeration in `tensor_basis` order with multiplication maps.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct MonomialTable {
    pub exponents: Vec<[u16; 4]>,
    /// `degree_start[d]..degree_start[d + 1]` holds the degree-`d` block.
    pub degree_start: Vec<usize>,
    /// Index after multiplying by `z_a`; `u32::MAX` beyond the maximal degree.
    pub multiply: Vec<[u32; 4]>,
    /// `e!` for each monomial (exact integers: no rank bound).
    pub factorials: Vec<Integer>,
    /// Metric expansion of the untranslated moment of `e`:
    /// `L(e) = sum (src, lambda, c) c g_lambda(src)`, `src = e - 2a`,
    /// `lambda = |a|`, `c = e! (-1)^(a1+a2+a3) / (2^|a| a!)`.
    pub qexp: Vec<Vec<(u32, u16, Rational)>>,
    /// Per axis: lines of monomials with fixed other exponents, as
    /// `(sum of the other exponents, indices for exponent 0, 1, ...)`.
    pub lines: [Vec<(u16, Vec<u32>)>; 4],
    /// Monomial degrees.
    pub degrees: Vec<u16>,
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
        let factorials: Vec<Integer> = exponents
            .iter()
            .map(|e| e.iter().fold(Integer::from(1), |product, &x| product * &factorial(x)))
            .collect();
        let qexp = exponents
            .iter()
            .enumerate()
            .map(|(index, e)| {
                let mut terms = Vec::new();
                for a0 in 0..=e[0] / 2 {
                    for a1 in 0..=e[1] / 2 {
                        for a2 in 0..=e[2] / 2 {
                            for a3 in 0..=e[3] / 2 {
                                let a = [a0, a1, a2, a3];
                                let src = [e[0] - 2 * a0, e[1] - 2 * a1, e[2] - 2 * a2, e[3] - 2 * a3];
                                let lambda: u16 = a.iter().sum();
                                let sign = if (a1 + a2 + a3) % 2 == 0 { 1 } else { -1 };
                                let denominator = a
                                    .iter()
                                    .fold(Integer::from(2).pow(u64::from(lambda)), |product, &x| product * &factorial(x));
                                let constant = Rational::from((
                                    Integer::from(sign) * &factorials[index],
                                    denominator,
                                ));
                                terms.push((lookup[&src], lambda, constant));
                            }
                        }
                    }
                }
                terms
            })
            .collect();
        let mut lines: [Vec<(u16, Vec<u32>)>; 4] = Default::default();
        for (axis, axis_lines) in lines.iter_mut().enumerate() {
            for (index, e) in exponents.iter().enumerate() {
                if e[axis] != 0 {
                    continue;
                }
                let others: u16 = e.iter().sum();
                let mut line = vec![index as u32];
                let mut next = *e;
                loop {
                    next[axis] += 1;
                    match lookup.get(&next) {
                        Some(&i) => line.push(i),
                        None => break,
                    }
                }
                axis_lines.push((others, line));
            }
        }
        let degrees = exponents.iter().map(|e| e.iter().sum()).collect();
        Self {
            exponents,
            degree_start,
            multiply,
            factorials,
            qexp,
            lines,
            degrees,
        }
    }

    pub fn block(&self, degree: usize) -> std::ops::Range<usize> {
        self.degree_start[degree]..self.degree_start[degree + 1]
    }
}


/// Post-processing of one accumulated instruction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PostOp {
    None,
    /// Multiply by `(a - 2 epsilon)^-1`, with `1/a` stored as a constant.
    DivideDimension { inverse: u32 },
    /// Partial fraction of a linearly dependent subset (moment programs with
    /// a Taylor jet only): Taylor order `k` of the destination is the terms
    /// applied to order `k + 1` of their sources, and the jet's last order
    /// is zero.
    TaylorShift,
}

/// Instructions and values of a Gram-free pentagon's coefficients.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PentagonBlock {
    pub instructions: (u32, u32),
    pub start: u32,
    pub count: u32,
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
    /// Linearly dependent propagators, `sum_i c_i D_i = 0` (the constant of
    /// the denominator identity vanishes): the Coulomb-exchange integrals
    /// of a heavy pair with equal momenta, `2 D_g = D_c + D_c~`. With the
    /// raised member `j`, `1 = -sum_(i != j) (c_i / c_j) D_i / D_j`, so every
    /// moment is `sum_(i != j) w_i` times the first Taylor coefficient, in the
    /// squared mass of `j`, of the pinch without `i` (that pinch with `D_j`
    /// doubled). Index into `DirectPlanData::homogeneous`.
    Homogeneous(usize),
}

/// A minimal affinely dependent propagator subset of a plan variant.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct RelationGeometry {
    pub mask: SubsetMask,
    pub members: Vec<usize>,
    /// Geometry offset of the identity weights, one per member.
    pub weights: u32,
}

/// A minimal linearly dependent propagator subset of a plan variant.
#[derive(Clone, Debug)]
pub(crate) struct HomogeneousGeometry {
    pub mask: SubsetMask,
    pub members: Vec<usize>,
    /// The member whose power the partial fraction raises (the plan's jet).
    pub raised: usize,
    /// Geometry offset of the weights `-c_i / c_raised`, one per member
    /// (the raised member's entry is unused).
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
    /// Gram-free part: `X^-1` (5x5, row-major, index 0 = the m0 row) of the
    /// modified Cayley matrix `X = [[2 m0^2, f^T], [f, Z]]`, and
    /// `Xtt_(kn)(0i) / det X = X^-1_0n X^-1_ik - X^-1_in X^-1_0k` at
    /// `adjadj + 16 (k-1) + 4 (n-1) + (i-1)`, `k, n, i = 1..4`.
    pub xinv: u32,
    pub adjadj: u32,
    /// Sums that collect the D(0) terms: `sum_{n=1..4} X^-1_n0`, then for
    /// k = 1..4 `sum_{n=0..4} X^-1_kn`, then for (k, i) the sums over
    /// `n != k` of `Xtt_(kn)(0i)/det X` at `sums + 5 + 4 (k-1) + (i-1)`.
    pub sums: u32,
    /// Coefficient ranks `0..=dd_rank` come from the Gram-free reduction;
    /// `-1` keeps the Gram recursion for every rank (dotted propagators,
    /// degenerate box children).
    pub dd_rank: i32,
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
    /// Per kappa: its momentum product in `DirectPlanData::products`.
    pub products: Vec<u32>,
}

/// One momentum product `P_kappa(z) = prod_i (p_i.z)^(kappa_i) / kappa_i!`,
/// shared by every gather whose subset has the same origin: their momenta are
/// all `shift(member) - shift(origin)`, and each kappa extends its parent at
/// its last nonzero momentum, so those gathers multiply the same vectors in
/// the same order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Product {
    pub degree: u16,
    /// Parent product; `u32::MAX` for degree zero.
    pub parent: u32,
    /// Geometry offset of the four components of the momentum multiplied in.
    pub momentum: u32,
    /// Exponent of that momentum (the `1/kappa!` divisor).
    pub divisor: u16,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) struct Subset {
    pub mask: SubsetMask,
    pub members: Vec<usize>,
    pub kind: SubsetKind,
    pub coefficient_rank: i32,
    pub moment_rank: i32,
    pub geometry_base: u32,
    pub pv: Option<PvGeometry>,
    pub pentagon: Option<PentagonGeometry>,
    /// Thin weights `w[column][i]` of a six- or seven-point subset, 5 columns
    /// x `members.len()` (a seven-point lane gives one member weight zero).
    pub hexagon: Option<u32>,
    /// Value index receiving the scalar master, for PV subsets.
    pub master: Option<u32>,
    /// Value index of the first moment, in `tensor_basis` order.
    pub moment_base: Option<u32>,
    /// Lightlike (or coincident) bubble: coefficients are closed forms
    /// filled per lane instead of a singular Gram recursion.
    pub lightlike: bool,
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
    /// Momentum products of all gathers, parents first.
    pub products: Vec<Product>,
    /// Pentagon and hexagon moment recursions, executed after gathers.
    pub moment_program: Vec<Instruction>,
    pub terms: Vec<Term>,
    /// Masters per family `A0..D0`: `(subset, value index)`.
    pub masters: [Vec<(usize, u32)>; 4],
    /// Minimal degenerate subsets of this variant, in profile order.
    pub relations: Vec<RelationGeometry>,
    /// Minimal linearly dependent subsets of this variant (Coulomb exchange).
    pub homogeneous: Vec<HomogeneousGeometry>,
    /// Lightlike bubbles: `(subset, [(lambda, k, value index)])` excluding
    /// the master `(0, 0)`.
    pub lightlike: Vec<(usize, Vec<(u16, u16, u32)>)>,
    pub output: Vec<u32>,
    pub gather_scratch: usize,
    /// Differentiated propagator and Taylor length `K` (its power). The
    /// output is the order-`K-1` Taylor coefficient in its squared mass,
    /// except in a Coulomb-exchange variant (see `output_order`).
    pub jet: Option<(usize, usize)>,
    /// Taylor order of the output: `K - 1` for a raised power, 0 otherwise
    /// (a Coulomb-exchange variant differentiates its pinches only).
    pub output_order: usize,
    /// Gathers read `C(lambda + h, kappa)` and scale by `(-2)^h`.
    pub lambda_shift: u16,
    /// Value index holding the constant `[0, 0, 0, 1]` (unit finite part),
    /// used by the rank-5 pentagon UV remainder.
    pub unit_finite: Option<u32>,
    /// Per subset: its coefficient instructions `coefficient_program[a..b]`
    /// (PV subsets; empty otherwise) and the number of its coefficient values
    /// `C(lambda, kappa)`, contiguous from its master. A refused box or
    /// triangle can thus be computed elsewhere and injected (partial rescue).
    pub coefficient_ranges: Vec<(u32, u32)>,
    pub coefficient_counts: Vec<u32>,
    /// Per subset, for Gram-free pentagons: their instructions
    /// `coefficient_program[a..b]` and their values, contiguous from `start`
    /// (`E(lambda, kappa)` and the shifted box `D(0)` interleaved), which no
    /// other subset's instructions write (results shared between the
    /// integrals of an amplitude).
    pub pentagon_blocks: Vec<Option<PentagonBlock>>,
    /// Per subset: its sub-subsets' kinds, coefficient ranks and lightlike
    /// flags in this plan (absent ones too), by their members in the
    /// subset's order, and the plan's dimension shift. These decide the
    /// gates, injections and closed forms beneath a subset, so equal
    /// signatures with equal kinematics and executor settings give equal
    /// coefficients (shared results' keys).
    pub subset_signatures: Vec<u64>,
    /// Per subset: the terms of its coefficient instructions (those of its
    /// `coefficient_ranges` entry, or of its pentagon block).
    pub coefficient_terms: Vec<u32>,
    /// Per PV subset mask: value index of `C(lambda, kappa)` (the small-Gram
    /// expansion fills a refused box from its triangles through these).
    pub coefficient_lookup: FastMap<SubsetMask, FastMap<(u16, Vec<u16>), u32>>,
    /// Extra coefficient ranks of the top subset's pinches (see
    /// `with_top_child_boost`); variants keep it.
    pub top_child_boost: u16,
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

/// `count + by` for the plan's `u32` value and geometry indices; a plan
/// beyond them (memory binds long before) is refused, never wrapped.
fn grow(count: u32, by: u32) -> Result<u32, ReductionError> {
    count.checked_add(by).ok_or_else(|| {
        ReductionError::Unsupported("direct plan: more values or geometry entries than its u32 indices address".into())
    })
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

fn without(mask: SubsetMask, member: usize) -> SubsetMask {
    mask & !((1 as SubsetMask) << member)
}

pub(crate) fn members(mask: SubsetMask) -> Vec<usize> {
    let mut rest = mask;
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while rest != 0 {
        out.push(rest.trailing_zeros() as usize);
        rest &= rest - 1;
    }
    out
}

/// Per-PV-subset coefficient layout.
struct CoefficientLayout {
    lookup: FastMap<(u16, Vec<u16>), u32>,
    spz: HashMap<(u16, Vec<u16>), u32>,
}

impl DirectPlanData {
    pub fn supported(settings: &ReductionSettings) -> Result<(), ReductionError> {
        // The structural bounds do not depend on the carrier.
        DirectCapabilities::of_representation(53).admits(settings)
    }

    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::with_relations(settings, &[])
    }

    /// Plan variant in which every subset (of at most five propagators)
    /// containing one of `relation_masks` is reduced by that subset's
    /// constant-one denominator identity instead of its singular Gram system.
    pub fn with_relations(
        settings: ReductionSettings,
        relation_masks: &[SubsetMask],
    ) -> Result<Self, ReductionError> {
        Self::with_profile(settings, relation_masks, &[])
    }

    /// Plan variant with denominator identities for `relation_masks` and
    /// closed-form coefficients for the lightlike bubbles `lightlike_masks`.
    pub fn with_profile(
        settings: ReductionSettings,
        relation_masks: &[SubsetMask],
        lightlike_masks: &[SubsetMask],
    ) -> Result<Self, ReductionError> {
        Self::with_profile_boost(settings, relation_masks, lightlike_masks, 0)
    }

    /// Plan whose top subset's pinches carry `boost` more coefficient ranks
    /// than the top's recursion needs: the small-Gram expansion of a box
    /// reaches its rank-`r` coefficients from triangles of rank `r + boost`.
    pub fn with_top_child_boost(settings: ReductionSettings, boost: u16) -> Result<Self, ReductionError> {
        Self::with_profile_boost(settings, &[], &[], boost)
    }

    pub fn with_profile_boost(
        settings: ReductionSettings,
        relation_masks: &[SubsetMask],
        lightlike_masks: &[SubsetMask],
        boost: u16,
    ) -> Result<Self, ReductionError> {
        Self::with_full_profile(settings, relation_masks, lightlike_masks, &[], boost)
    }

    /// Plan variant with constant-one identities for `relation_masks`,
    /// lightlike bubbles `lightlike_masks` and partial fractions for the
    /// linearly dependent sets `homogeneous` (mask, raised member). Subsets
    /// containing a constant relation take it first.
    pub fn with_full_profile(
        settings: ReductionSettings,
        relation_masks: &[SubsetMask],
        lightlike_masks: &[SubsetMask],
        homogeneous: &[(SubsetMask, usize)],
        boost: u16,
    ) -> Result<Self, ReductionError> {
        Self::supported(&settings)?;
        let lambda_shift = settings.dimension_shift as u16;
        let raised = homogeneous.first().map(|&(_, j)| j);
        if let Some(j) = raised {
            if settings.powers.iter().any(|&p| p != 1) {
                return Err(ReductionError::Unsupported(
                    "direct engine: linearly dependent propagators with a raised power".into(),
                ));
            }
            if homogeneous.iter().any(|&(_, r)| r != j) {
                return Err(ReductionError::Unsupported(
                    "direct engine: linearly dependent propagator sets raising different propagators".into(),
                ));
            }
            let mut dotted = settings.clone();
            dotted.powers[j] = 2;
            Self::supported(&dotted)?;
        }
        let jet = match raised {
            Some(j) => Some((j, 2)),
            None => settings
                .powers
                .iter()
                .position(|&p| p > 1)
                .map(|d| (d, usize::from(settings.powers[d]))),
        };
        let output_order = if raised.is_some() { 0 } else { jet.map_or(0, |(_, k)| k - 1) };
        let relation_for = |mask: SubsetMask| -> Option<usize> {
            if mask.count_ones() >= 6 {
                return None;
            }
            relation_masks.iter().position(|&r| r & !mask == 0)
        };
        let homogeneous_for = |mask: SubsetMask| -> Option<usize> {
            if mask.count_ones() >= 6 || relation_for(mask).is_some() {
                return None;
            }
            homogeneous.iter().position(|&(h, _)| h & !mask == 0)
        };
        let n = settings.propagators;
        let rank = settings.rank;
        let basis = tensor_basis(rank);
        let monomials = MonomialTable::new(rank);
        let top = full_mask(n);
        // Gram-free pentagon rank: not for dotted propagators (no geometry
        // jets of X^-1 yet) or when a box child is degenerate (it supplies
        // moments by its identity, not PV coefficients).
        let dd_rank = |mask: SubsetMask, moment: i32| -> i32 {
            if moment < 0 || jet.is_some() {
                return -1;
            }
            if members(mask).into_iter().any(|member| relation_for(without(mask, member)).is_some()) {
                return -1;
            }
            moment.min(DD_PENTAGON_RANK)
        };

        // Propagate rank needs from the top topology to its pinches.
        let mut coefficient_need: HashMap<SubsetMask, i32> = HashMap::new();
        let mut moment_need: HashMap<SubsetMask, i32> = HashMap::new();
        moment_need.insert(top, i32::from(rank));
        // Every nonempty subset: memory, not a cap, bounds the point count.
        let mut masks: Vec<SubsetMask> = Vec::new();
        masks.try_reserve_exact(top as usize).map_err(|_| {
            ReductionError::Unsupported(format!(
                "direct engine: the 2^{n} - 1 subsets of a {n}-propagator plan exceed the available memory"
            ))
        })?;
        masks.extend(1..=top);
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
            if let Some(h) = homogeneous_for(mask) {
                if moment >= 0 {
                    let (h_mask, j) = homogeneous[h];
                    for member in members(h_mask).into_iter().filter(|&member| member != j) {
                        let child = without(mask, member);
                        let need = moment_need.entry(child).or_insert(-1);
                        *need = (*need).max(moment);
                    }
                }
                continue;
            }
            match size {
                5.. => {
                    let dd = if size == 5 { dd_rank(mask, moment) } else { -1 };
                    if moment > dd {
                        for member in members(mask) {
                            let child = without(mask, member);
                            let need = moment_need.entry(child).or_insert(-1);
                            *need = (*need).max((moment - 1).max(0));
                        }
                    }
                    if dd >= 0 {
                        // Box coefficients up to the same rank (Denner-Dittmaier).
                        for member in members(mask) {
                            let child = without(mask, member);
                            let need = coefficient_need.entry(child).or_insert(-1);
                            *need = (*need).max(dd);
                        }
                    }
                }
                _ => {
                    if moment >= 0 {
                        let need = coefficient_need.entry(mask).or_insert(-1);
                        *need = (*need).max(moment + 2 * i32::from(lambda_shift));
                    }
                    let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
                    if coefficient >= 1 && size >= 2 {
                        let extra = if mask == top { i32::from(boost) } else { 0 };
                        for member in members(mask) {
                            let child = without(mask, member);
                            let need = coefficient_need.entry(child).or_insert(-1);
                            *need = (*need).max(coefficient - 1 + extra);
                        }
                    }
                }
            }
        }

        // Coefficient ranks are u16 (`DirectCapabilities::max_coefficient_rank`):
        // `supported` admits R + 2h within them; a top-child boost on top
        // must fit as well.
        let largest = coefficient_need.values().copied().max().unwrap_or(-1);
        if largest > i32::from(u16::MAX) {
            return Err(ReductionError::Unsupported(format!(
                "direct plan: coefficient rank {largest} beyond its u16 coefficient ranks"
            )));
        }

        // Mass derivatives of a master reach every pinch of it by IBP.
        if let Some((dotted, _)) = jet {
            let parents: Vec<SubsetMask> = coefficient_need
                .iter()
                .filter(|&(&mask, &need)| need >= 0 && mask & (1 << dotted) != 0)
                .map(|(&mask, _)| mask)
                .collect();
            for parent in parents {
                let mut sub = parent;
                while sub != 0 {
                    let need = coefficient_need.entry(sub).or_insert(-1);
                    *need = (*need).max(0);
                    sub = (sub - 1) & parent;
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
        let mut subset_index: HashMap<SubsetMask, usize> = HashMap::new();
        let mut value_count: u32 = 0;
        let mut geometry_count: u32 = 0;
        let mut layouts: HashMap<SubsetMask, CoefficientLayout> = HashMap::new();
        // D(0) of a Gram-free pentagon in its own momentum basis.
        let mut shifted_layouts: HashMap<SubsetMask, HashMap<(u16, Vec<u16>), u32>> = HashMap::new();
        let mut unit_finite = None;
        let mut relations = Vec::new();
        for &relation in relation_masks {
            let member_list = members(relation);
            relations.push(RelationGeometry {
                mask: relation,
                weights: geometry_count,
                members: member_list.clone(),
            });
            geometry_count = grow(geometry_count, member_list.len() as u32)?;
        }
        let mut homogeneous_geometry = Vec::new();
        for &(h_mask, j) in homogeneous {
            let member_list = members(h_mask);
            homogeneous_geometry.push(HomogeneousGeometry {
                mask: h_mask,
                weights: geometry_count,
                members: member_list.clone(),
                raised: j,
            });
            geometry_count = grow(geometry_count, member_list.len() as u32)?;
        }
        for &mask in &masks {
            let coefficient = coefficient_need.get(&mask).copied().unwrap_or(-1);
            let moment = moment_need.get(&mask).copied().unwrap_or(-1);
            if coefficient < 0 && moment < 0 {
                continue;
            }
            let member_list = members(mask);
            let size = member_list.len();
            let kind = match (relation_for(mask), homogeneous_for(mask), size) {
                (Some(relation), _, _) => SubsetKind::Relation(relation),
                (None, Some(h), _) => SubsetKind::Homogeneous(h),
                (None, None, 6..) => SubsetKind::Hexagon,
                (None, None, 5) => SubsetKind::Pentagon,
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
                lightlike: size == 2 && lightlike_masks.contains(&mask),
            };
            match kind {
                SubsetKind::Pv => {
                    let (geometry, length) = PvGeometry::new(geometry_base, size - 1);
                    subset.pv = Some(geometry);
                    geometry_count = grow(geometry_count, length)?;
                    let momenta = size - 1;
                    let mut layout = CoefficientLayout {
                        lookup: FastMap::default(),
                        spz: HashMap::new(),
                    };
                    for r in 0..=coefficient.max(0) as u16 {
                        for lambda in (0..=r / 2).rev() {
                            for kappa in compositions(r - 2 * lambda, momenta) {
                                layout.lookup.insert((lambda, kappa), value_count);
                                value_count = grow(value_count, 1)?;
                            }
                        }
                    }
                    if size >= 2 && coefficient >= 1 {
                        for r in 0..coefficient as u16 {
                            for lambda in (0..=r / 2).rev() {
                                for kappa in compositions(r - 2 * lambda, momenta) {
                                    layout.spz.insert((lambda, kappa), value_count);
                                    value_count = grow(value_count, 1)?;
                                }
                            }
                        }
                    }
                    subset.master = Some(layout.lookup[&(0, vec![0; momenta])]);
                    layouts.insert(mask, layout);
                }
                SubsetKind::Pentagon => {
                    let dd = dd_rank(mask, moment);
                    let mut pentagon = PentagonGeometry {
                        w: geometry_base,
                        a: geometry_base + 16,
                        s: geometry_base + 20,
                        b: geometry_base + 24,
                        xinv: NO_GEOMETRY,
                        adjadj: NO_GEOMETRY,
                        sums: NO_GEOMETRY,
                        dd_rank: dd,
                    };
                    geometry_count = grow(geometry_count, 29)?;
                    if dd >= 0 {
                        // Momenta and origin for the gather, X^-1, Xtt/det X.
                        let (geometry, length) = PvGeometry::new(geometry_count, 4);
                        subset.pv = Some(geometry);
                        geometry_count = grow(geometry_count, length)?;
                        pentagon.xinv = geometry_count;
                        geometry_count = grow(geometry_count, 25)?;
                        pentagon.adjadj = geometry_count;
                        geometry_count = grow(geometry_count, 64)?;
                        pentagon.sums = geometry_count;
                        geometry_count = grow(geometry_count, 21)?;
                        subset.coefficient_rank = dd;
                        let mut layout = CoefficientLayout {
                            lookup: FastMap::default(),
                            spz: HashMap::new(),
                        };
                        let mut shifted = HashMap::new();
                        for r in 0..=dd as u16 {
                            for lambda in (0..=r / 2).rev() {
                                for kappa in compositions(r - 2 * lambda, 4) {
                                    layout.lookup.insert((lambda, kappa.clone()), value_count);
                                    shifted.insert((lambda, kappa), value_count + 1);
                                    value_count = grow(value_count, 2)?;
                                }
                            }
                        }
                        layouts.insert(mask, layout);
                        shifted_layouts.insert(mask, shifted);
                        if dd >= 5 && unit_finite.is_none() {
                            unit_finite = Some(value_count);
                            value_count = grow(value_count, 1)?;
                        }
                    }
                    subset.pentagon = Some(pentagon);
                }
                SubsetKind::Hexagon => {
                    subset.hexagon = Some(geometry_base);
                    geometry_count = grow(geometry_count, 5 * size as u32)?;
                }
                SubsetKind::Relation(_) | SubsetKind::Homogeneous(_) => {
                    if coefficient >= 0 {
                        return Err(ReductionError::Preparation(
                            "a degenerate subset cannot supply PV coefficients".into(),
                        ));
                    }
                }
            }
            if moment >= 0 {
                subset.moment_base = Some(value_count);
                value_count = grow(value_count, tensor_basis(moment as u16).len() as u32)?;
            }
            subset_index.insert(mask, subsets.len());
            subsets.push(subset);
        }

        // Coefficient programs: increasing subset size, then rank.
        let mut coefficient_program = Vec::new();
        let mut masters: [Vec<(usize, u32)>; 4] = Default::default();
        let mut lightlike = Vec::new();
        let mut range_starts = Vec::with_capacity(subsets.len() + 1);
        for (id, subset) in subsets.iter().enumerate() {
            range_starts.push(coefficient_program.len() as u32);
            if subset.kind != SubsetKind::Pv {
                continue;
            }
            let size = subset.members.len();
            masters[size - 1].push((id, subset.master.unwrap()));
            let geometry = subset.pv.unwrap();
            let layout = &layouts[&subset.mask];
            let momenta = size - 1;
            let c = subset.coefficient_rank.max(0) as u16;
            if subset.lightlike {
                let mut entries = Vec::new();
                for r in 1..=c {
                    for lambda in 0..=r / 2 {
                        let k = r - 2 * lambda;
                        entries.push((lambda, k, layout.lookup[&(lambda, vec![k])]));
                    }
                }
                lightlike.push((id, entries));
                continue;
            }
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

        range_starts.push(coefficient_program.len() as u32);
        let coefficient_ranges: Vec<(u32, u32)> =
            (0..subsets.len()).map(|id| (range_starts[id], range_starts[id + 1])).collect();
        let coefficient_counts: Vec<u32> = subsets
            .iter()
            .map(|subset| match (subset.kind, layouts.get(&subset.mask)) {
                (SubsetKind::Pv, Some(layout)) => layout.lookup.len() as u32,
                _ => 0,
            })
            .collect();
        // Gram-free pentagon coefficients (Denner--Dittmaier, hep-ph/0509141
        // Eqs. (5.9)-(5.10), symmetrized as in (5.12)): with I = kappa - e_k,
        //   R E_{lambda,kappa} = 2 lambda Ebar0 + sum_k kappa_k Ebar_k(I),
        //   Ebar0 = sum_n X^-1_n0 [D_{lambda,kappa}(n) - D_{lambda,kappa}(0)],
        //   Ebar_k = sum_n X^-1_kn [D_I(n) - D_I(0)] - X^-1_k0 D_I(0)
        //     - 2 sum_n sum_i I_i Xtt_(kn)(0i)/det X [D_{lambda+1,I-e_i}(n) - .(0)],
        // with the box coefficients of rank <= R only (no Gram inverse); the
        // rank-5 UV remainder replaces D_0000(0) by D_0000(0) + 1/48 in
        // E_{0000k}. D(n), n >= 1, drops momentum n (zero if I_n > 0); D(0)
        // is the box without member 0 in this basis (D_{kappa}(0) = Dtilde for
        // kappa_1 = 0, else -sum_{s=2..4} D_{kappa-e1+es}(0) - D_{kappa-e1}(0)).
        let mut pentagon_blocks: Vec<Option<PentagonBlock>> = vec![None; subsets.len()];
        for (id, subset) in subsets.iter().enumerate() {
            let Some(geometry) = subset.pentagon else { continue };
            if subset.kind != SubsetKind::Pentagon || geometry.dd_rank < 0 {
                continue;
            }
            let first_instruction = coefficient_program.len() as u32;
            let dd = geometry.dd_rank as u16;
            let layout = &layouts[&subset.mask];
            let shifted = &shifted_layouts[&subset.mask];
            let child = |n: usize| without(subset.mask, subset.members[n]);
            let box_coefficient = |n: usize, lambda: u16, kappa: &[u16]| -> Option<u32> {
                if kappa[n - 1] != 0 {
                    return None;
                }
                let mut reduced = kappa.to_vec();
                reduced.remove(n - 1);
                Some(layouts[&child(n)].lookup[&(lambda, reduced)])
            };
            let zero_layout = &layouts[&child(0)];
            for r in 0..=dd {
                for lambda in (0..=r / 2).rev() {
                    let mut kappas = compositions(r - 2 * lambda, 4);
                    kappas.sort_by_key(|kappa| kappa[0]);
                    for kappa in kappas {
                        let dst = shifted[&(lambda, kappa.clone())];
                        let terms = if kappa[0] == 0 {
                            let src = zero_layout.lookup[&(lambda, kappa[1..].to_vec())];
                            vec![builder.term(src, NO_GEOMETRY, rational(1, 1))]
                        } else {
                            let mut lower = kappa.clone();
                            lower[0] -= 1;
                            let mut terms =
                                vec![builder.term(shifted[&(lambda, lower.clone())], NO_GEOMETRY, rational(-1, 1))];
                            for s in 1..4 {
                                let mut other = lower.clone();
                                other[s] += 1;
                                terms.push(builder.term(shifted[&(lambda, other)], NO_GEOMETRY, rational(-1, 1)));
                            }
                            terms
                        };
                        coefficient_program.push(builder.instruction(dst, terms, PostOp::None));
                    }
                }
            }
            let xinv = |i: usize, j: usize| geometry.xinv + (5 * i + j) as u32;
            for r in 0..=dd {
                let big_r = i64::from(r);
                for lambda in (0..=r / 2).rev() {
                    for kappa in compositions(r - 2 * lambda, 4) {
                        let dst = layout.lookup[&(lambda, kappa.clone())];
                        let mut terms = Vec::new();
                        if r == 0 {
                            // Scalar pentagon: Melrose, b = Y^-1 1.
                            for i in 0..5 {
                                let src = layouts[&child(i)].lookup[&(0, vec![0, 0, 0])];
                                terms.push(builder.term(src, geometry.b + i as u32, rational(1, 1)));
                            }
                            coefficient_program.push(builder.instruction(dst, terms, PostOp::None));
                            continue;
                        }
                        if lambda >= 1 {
                            let c = rational(2 * i64::from(lambda), big_r);
                            let d0 = shifted[&(lambda, kappa.clone())];
                            for n in 1..=4 {
                                if let Some(src) = box_coefficient(n, lambda, &kappa) {
                                    terms.push(builder.term(src, xinv(n, 0), c.clone()));
                                }
                            }
                            terms.push(builder.term(d0, geometry.sums, -c.clone()));
                        }
                        for k in 1..=4 {
                            if kappa[k - 1] == 0 {
                                continue;
                            }
                            let weight = i64::from(kappa[k - 1]);
                            let c = rational(weight, big_r);
                            let mut rest = kappa.clone();
                            rest[k - 1] -= 1;
                            let d0 = shifted[&(lambda, rest.clone())];
                            for n in 1..=4 {
                                if let Some(src) = box_coefficient(n, lambda, &rest) {
                                    terms.push(builder.term(src, xinv(k, n), c.clone()));
                                }
                            }
                            terms.push(builder.term(d0, geometry.sums + k as u32, -c.clone()));
                            for i in 1..=4 {
                                if rest[i - 1] == 0 {
                                    continue;
                                }
                                let ci = rational(-2 * weight * i64::from(rest[i - 1]), big_r);
                                let mut lower = rest.clone();
                                lower[i - 1] -= 1;
                                let d0 = shifted[&(lambda + 1, lower.clone())];
                                for n in 1..=4 {
                                    if n == k {
                                        continue;
                                    }
                                    let g = geometry.adjadj + (16 * (k - 1) + 4 * (n - 1) + (i - 1)) as u32;
                                    if let Some(src) = box_coefficient(n, lambda + 1, &lower) {
                                        terms.push(builder.term(src, g, ci.clone()));
                                    }
                                }
                                let g = geometry.sums + (5 + 4 * (k - 1) + (i - 1)) as u32;
                                terms.push(builder.term(d0, g, -ci.clone()));
                            }
                            if r == 5 && lambda == 2 {
                                let unit = unit_finite.expect("rank-5 pentagons allocate the unit value");
                                terms.push(builder.term(unit, xinv(k, 0), rational(-weight, 48 * big_r)));
                            }
                        }
                        coefficient_program.push(builder.instruction(dst, terms, PostOp::None));
                    }
                }
            }
            pentagon_blocks[id] = Some(PentagonBlock {
                instructions: (first_instruction, coefficient_program.len() as u32),
                start: layout.lookup[&(0, vec![0; 4])],
                count: 2 * layout.lookup.len() as u32,
            });
        }

        let subset_signatures: Vec<u64> = subsets
            .iter()
            .map(|subset| {
                let mut signature: u64 = 0xcbf2_9ce4_8422_2325;
                let mut eat = |x: u64| {
                    signature ^= x;
                    signature = signature.wrapping_mul(0x0100_0000_01b3);
                };
                for relative in 1u64..=(u64::MAX >> (64 - subset.members.len())) {
                    let mask = subset
                        .members
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| relative & (1 << i) != 0)
                        .fold(0 as SubsetMask, |mask, (_, &member)| mask | ((1 as SubsetMask) << member));
                    match subset_index.get(&mask).map(|&k| &subsets[k]) {
                        None => eat(u64::MAX),
                        Some(sub) => {
                            let kind = match sub.kind {
                                SubsetKind::Pv => 0,
                                SubsetKind::Pentagon => 1,
                                SubsetKind::Hexagon => 2,
                                SubsetKind::Relation(_) => 3,
                                SubsetKind::Homogeneous(_) => 4,
                            };
                            eat(kind | u64::from(sub.lightlike) << 3);
                            eat(sub.coefficient_rank as i64 as u64);
                        }
                    }
                }
                eat(u64::from(lambda_shift));
                signature
            })
            .collect();

        let coefficient_terms: Vec<u32> = (0..subsets.len())
            .map(|id| {
                let (a, b) = pentagon_blocks[id].map_or(coefficient_ranges[id], |block| block.instructions);
                coefficient_program[a as usize..b as usize].iter().map(|i| i.terms.1 - i.terms.0).sum()
            })
            .collect();

        // Gathers of PV subsets whose moments are needed.
        let mut gathers = Vec::new();
        let mut products: Vec<Product> = Vec::new();
        // (origin, (propagator, exponent) of each member momentum with a
        // nonzero exponent, in propagator order) -> product: sized by the
        // subset, so any propagator index is addressed.
        let mut product_lookup: HashMap<(usize, Vec<(usize, u16)>), u32> = HashMap::new();
        let mut gather_scratch = 0usize;
        for (id, subset) in subsets.iter().enumerate() {
            let gram_free = subset.pentagon.is_some_and(|geometry| geometry.dd_rank >= 0);
            if !(subset.kind == SubsetKind::Pv || gram_free) || subset.moment_rank < 0 {
                continue;
            }
            let momenta = subset.members.len() - 1;
            let rank = if gram_free {
                subset.moment_rank.min(subset.coefficient_rank) as u16
            } else {
                subset.moment_rank as u16
            };
            let layout = &layouts[&subset.mask];
            let pv = subset.pv.unwrap();
            let origin = subset.members[0];
            let embed = |kappa: &[u16]| {
                let exponents: Vec<(usize, u16)> = kappa
                    .iter()
                    .enumerate()
                    .filter(|&(_, &k)| k != 0)
                    .map(|(i, &k)| (subset.members[i + 1], k))
                    .collect();
                (origin, exponents)
            };
            let mut kappas = Vec::new();
            let mut kappa_products = Vec::new();
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
                    let key = embed(&kappa);
                    let product = match product_lookup.get(&key) {
                        Some(&product) => product,
                        None => {
                            let product = products.len() as u32;
                            products.push(Product {
                                degree,
                                parent: if degree == 0 { u32::MAX } else { kappa_products[parent as usize] },
                                momentum: pv.p + 4 * u32::from(momentum),
                                divisor,
                            });
                            product_lookup.insert(key, product);
                            product
                        }
                    };
                    kappa_products.push(product);
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
                            layout.lookup[&(lambda + lambda_shift, kappa.clone())];
                    }
                }
            }
            gathers.push(Gather {
                subset: id,
                moment_rank: rank,
                kappas,
                coefficient,
                moment_base: subset.moment_base.unwrap(),
                products: kappa_products,
            });
        }

        // Pentagon and hexagon Cartesian moment recursions.
        let mut moment_program = Vec::new();
        let child_moment = |mask: SubsetMask, exponents: [u16; 4]| -> u32 {
            let child = &subsets[subset_index[&mask]];
            let rank: u16 = exponents.iter().sum();
            debug_assert!(i32::from(rank) <= child.moment_rank);
            child.moment_base.unwrap() + index_in_basis(exponents)
        };
        // Subsets whose Taylor orders above the last one the partial fraction
        // gives are missing: the linearly dependent subsets (order k from
        // order k + 1 of their pinches, so the jet's last order is zero) and
        // every moment recursion that reads one. A partial fraction reading
        // such a subset would miss that order.
        let mut truncated: HashSet<SubsetMask> = HashSet::new();
        for subset in &subsets {
            if subset.moment_rank < 0 {
                continue;
            }
            let read: Vec<SubsetMask> = match subset.kind {
                SubsetKind::Pv => Vec::new(),
                SubsetKind::Relation(relation) => {
                    relations[relation].members.iter().map(|&member| without(subset.mask, member)).collect()
                }
                SubsetKind::Homogeneous(h) => {
                    let geometry = &homogeneous_geometry[h];
                    geometry
                        .members
                        .iter()
                        .filter(|&&member| member != geometry.raised)
                        .map(|&member| without(subset.mask, member))
                        .collect()
                }
                SubsetKind::Pentagon | SubsetKind::Hexagon => {
                    subset.members.iter().map(|&member| without(subset.mask, member)).collect()
                }
            };
            if read.iter().any(|child| truncated.contains(child)) {
                if let SubsetKind::Homogeneous(_) = subset.kind {
                    return Err(ReductionError::Unsupported(format!(
                        "direct engine: linearly dependent propagators {:#b} contain a pinch that needs \
                         another linearly dependent set's partial fraction",
                        subset.mask
                    )));
                }
                truncated.insert(subset.mask);
            }
            match subset.kind {
                SubsetKind::Pv => {}
                SubsetKind::Homogeneous(h) => {
                    // Partial fraction of a linearly dependent subset, after
                    // its pinches and before every parent: its order-0
                    // moments are the order-1 Taylor coefficients of the
                    // pinches with a doubled massless member.
                    truncated.insert(subset.mask);
                    let geometry = &homogeneous_geometry[h];
                    let base = subset.moment_base.unwrap();
                    for (index, e) in tensor_basis(subset.moment_rank as u16).iter().enumerate() {
                        let terms = geometry
                            .members
                            .iter()
                            .enumerate()
                            .filter(|&(_, &member)| member != geometry.raised)
                            .map(|(j, &member)| {
                                builder.term(
                                    child_moment(without(subset.mask, member), e.0),
                                    geometry.weights + j as u32,
                                    rational(1, 1),
                                )
                            })
                            .collect();
                        moment_program.push(builder.instruction(base + index as u32, terms, PostOp::TaylorShift));
                    }
                }
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
                        if (e.rank() as i32) <= geometry.dd_rank {
                            // Gathered from the Gram-free coefficients.
                            continue;
                        }
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
                    let size = subset.members.len();
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
                                    weights + (size * column + i) as u32,
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

        // Terms, constants and instructions are addressed by u32 too.
        let limit = u32::MAX as usize;
        if builder.terms.len() > limit
            || builder.constants.len() > limit
            || coefficient_program.len() > limit
            || moment_program.len() > limit
        {
            return Err(ReductionError::Unsupported(
                "direct plan: more terms, constants or instructions than its u32 indices address".into(),
            ));
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
            products,
            moment_program,
            terms: builder.terms,
            masters,
            relations,
            homogeneous: homogeneous_geometry,
            lightlike,
            output,
            gather_scratch,
            jet,
            output_order,
            lambda_shift,
            unit_finite,
            coefficient_ranges,
            coefficient_counts,
            pentagon_blocks,
            subset_signatures,
            coefficient_terms,
            coefficient_lookup: layouts.into_iter().map(|(mask, layout)| (mask, layout.lookup)).collect(),
            top_child_boost: boost,
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

    /// A Gram-free pentagon's block (shared between the integrals of an
    /// amplitude) holds its coefficients and is written by its own
    /// instructions only; equal pentagons in different plans have equal
    /// signatures.
    #[test]
    fn pentagon_blocks_are_private_and_signed() {
        let mut signatures = Vec::new();
        for (n, rank) in [(5usize, 5u16), (6, 6), (7, 7)] {
            let plan = DirectPlanData::new(ReductionSettings::new(n, rank)).unwrap();
            let mut blocks = 0;
            for (id, block) in plan.pentagon_blocks.iter().enumerate() {
                let Some(block) = block else { continue };
                blocks += 1;
                let inside = |v: u32| (block.start..block.start + block.count).contains(&v);
                let (a, b) = block.instructions;
                for (index, instruction) in plan.coefficient_program.iter().enumerate() {
                    assert_eq!((a as usize..b as usize).contains(&index), inside(instruction.dst), "N{n} pentagon {id}");
                }
                assert!(plan.moment_program.iter().all(|instruction| !inside(instruction.dst)));
                let lookup = &plan.coefficient_lookup[&plan.subsets[id].mask];
                assert_eq!(2 * lookup.len() as u32, block.count);
                assert!(lookup.values().all(|&v| inside(v)));
                signatures.push(plan.subset_signatures[id]);
            }
            assert_eq!(blocks, [1, 6, 21][n - 5]);
        }
        assert!(signatures.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn basis_index_matches_enumeration() {
        for (index, monomial) in tensor_basis(7).iter().enumerate() {
            assert_eq!(index_in_basis(monomial.0) as usize, index, "{monomial:?}");
        }
    }

    #[test]
    fn plans_cover_one_to_seven_points() {
        for n in 1..=7 {
            for rank in 0..=6 {
                let plan = DirectPlanData::new(ReductionSettings::new(n, rank)).unwrap();
                assert_eq!(plan.output.len(), plan.basis.len());
                assert!(plan.output.iter().all(|&i| (i as usize) < plan.value_count));
            }
        }
    }

    /// Each gather's kappa reads a shared product built along the same chain
    /// (parent, divisor) from the same momentum `shift(member) - shift(origin)`.
    #[test]
    fn shared_products_follow_each_gathers_chain() {
        for n in 1..=7 {
            for rank in 0..=6 {
                let plan = DirectPlanData::new(ReductionSettings::new(n, rank)).unwrap();
                // Geometry offset of each gathered momentum -> (origin, member).
                let mut momenta = HashMap::new();
                for gather in &plan.gathers {
                    let subset = &plan.subsets[gather.subset];
                    let pv = subset.pv.unwrap();
                    for (i, &member) in subset.members[1..].iter().enumerate() {
                        momenta.insert(pv.p + 4 * i as u32, (subset.members[0], member));
                    }
                }
                let mut used = vec![false; plan.products.len()];
                for (index, product) in plan.products.iter().enumerate() {
                    assert!(product.degree == 0 || (product.parent as usize) < index);
                }
                for gather in &plan.gathers {
                    let subset = &plan.subsets[gather.subset];
                    assert_eq!(gather.products.len(), gather.kappas.len());
                    for (kappa, &index) in gather.kappas.iter().zip(&gather.products) {
                        used[index as usize] = true;
                        let product = plan.products[index as usize];
                        assert_eq!(product.degree, kappa.degree);
                        if kappa.degree == 0 {
                            continue;
                        }
                        assert_eq!(product.divisor, kappa.divisor);
                        assert_eq!(product.parent, gather.products[kappa.parent as usize]);
                        let member = subset.members[usize::from(kappa.momentum) + 1];
                        assert_eq!(momenta[&product.momentum], (subset.members[0], member), "N{n} rank {rank}");
                    }
                }
                assert!(used.into_iter().all(|used| used));
            }
        }
    }

    /// A partial fraction runs after its pinches and before every parent
    /// that reads it: here a relation parent (the pentagon of a planar lane,
    /// whose relation {0, 1, 2, 4} leaves the Coulomb box {1, 2, 3, 4}).
    #[test]
    fn partial_fractions_precede_the_recursions_that_read_them() {
        let plan = DirectPlanData::with_full_profile(
            ReductionSettings::new(5, 2),
            &[0b10111],
            &[],
            &[(0b01110, 2)],
            0,
        )
        .unwrap();
        let top = plan.subsets.last().unwrap();
        assert!(matches!(top.kind, SubsetKind::Relation(_)));
        let child = plan.subsets.iter().find(|subset| subset.mask == 0b11110).unwrap();
        assert!(matches!(child.kind, SubsetKind::Homogeneous(_)));
        let moments = |subset: &Subset| {
            let base = subset.moment_base.unwrap();
            base..base + tensor_basis(subset.moment_rank as u16).len() as u32
        };
        let (child_moments, top_moments) = (moments(child), moments(top));
        let written = plan
            .moment_program
            .iter()
            .rposition(|instruction| child_moments.contains(&instruction.dst))
            .unwrap();
        assert!(matches!(plan.moment_program[written].post, PostOp::TaylorShift));
        let read = plan
            .moment_program
            .iter()
            .position(|instruction| {
                plan.terms[instruction.terms.0 as usize..instruction.terms.1 as usize]
                    .iter()
                    .any(|term| child_moments.contains(&term.src))
            })
            .unwrap();
        assert!(top_moments.contains(&plan.moment_program[read].dst));
        assert!(written < read, "the relation reads the Coulomb box before its partial fraction");
    }

    /// A partial fraction whose pinch is itself a partial fraction would need
    /// that pinch's Taylor order 1, which no partial fraction gives: refused.
    #[test]
    fn nested_partial_fractions_are_refused() {
        let error = DirectPlanData::with_full_profile(
            ReductionSettings::new(5, 2),
            &[],
            &[],
            &[(0b00111, 1), (0b01110, 1)],
            0,
        )
        .unwrap_err();
        assert!(matches!(error, ReductionError::Unsupported(_)), "{error:?}");
    }
}
