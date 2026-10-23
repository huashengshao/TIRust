//! Symbolica compilation of the scalar dimension/power recurrences used by IREGI.
//!
//! The compiled quantity is `J_h = pi^(-h) I_(4+2h-2epsilon)`, with the common
//! `i*pi^2` and IREGI/OneLOop normalization conversion applied by the caller.
//! Recurrences are Eqs. (15), (18), (19), (21)--(23), (26), and (28) of
//! hep-ph/0303184,
//! as implemented in the reference `si_reduce.f90`. All expression operations
//! and matrix inversion are performed by Symbolica.
//!
//! A unit-power pentagon in `6-2epsilon` dimensions is not fixed by the
//! available A0--D0 Laurent coefficients through the finite term. The original
//! code completes such intermediate requests through a four-dimensional PV
//! tensor-coefficient path. Completing an ordinary four-dimensional tensor
//! through that path is valid. Its standalone shifted-scalar output disagrees
//! with independently converged defining-integral checks recorded in
//! `validation/shifted-pentagon/evidence.json`. Direct requests remain errors
//! by default. A caller with a qualified genuine D6 backend can explicitly
//! enable typed pentagon leaves. See the dimension-shift discussion and Eq. (6) of
//! [Kozlov--Lee](https://arxiv.org/html/1512.01165v2).

use std::{
    collections::HashMap,
    sync::{
        LazyLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::{atom::AtomField, integer::Integer},
    id::Replacement,
    tensors::matrix::Matrix,
};

/// A scalar integral in `4 + 2*dimension_shift - 2*epsilon` dimensions.
/// Zero propagator powers denote pinches; vector positions preserve routing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScalarRequest {
    pub powers: Vec<u16>,
    pub dimension_shift: i16,
}

/// The normalization and dimension of a scalar backend request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MasterLeafKind {
    /// Unit-power A0, B0, C0, or D0 in `4-2 epsilon` dimensions.
    FourDimensional,
    /// A genuine unit-power pentagon in `6-2 epsilon` dimensions.
    /// Its finite-domain qualification belongs to the numerical backend.
    SixDimensionalPentagon,
}

/// A unit-power scalar master, with routing preserved by active indices.
#[derive(Clone, Debug)]
pub struct MasterLeaf {
    pub kind: MasterLeafKind,
    pub active: Vec<usize>,
    /// `[UV single, IR single, IR double, finite]`, in OneLOop normalization.
    pub slots: [Atom; 4],
}

#[derive(Clone)]
struct Geometry {
    inverse: Vec<Vec<Atom>>,
    /// `b = R^-1 1`.
    b: Vec<Atom>,
    /// `B = 1^T R^-1 1`.
    sum_b: Atom,
}

#[derive(Clone)]
enum ExceptionalGeometry {
    InvertibleS(Vec<Vec<Atom>>),
    NullVector(Vec<Atom>),
}

#[derive(Clone, Copy)]
enum DegenerateFamily {
    Scaleless,
    Coincident(usize, usize),
    Distinct,
}

#[derive(Clone)]
struct RoutingRelation {
    coefficients: Vec<Atom>,
    /// The denominator combination equals one, or equals zero.
    constant_one: bool,
}

struct LightlikeBubbleExpansion {
    distance: Atom,
    mass_difference: Atom,
    derivative_order: u16,
    equal_masses: bool,
    base: Box<ScalarCompiler>,
    copied_aliases: usize,
}

static SCALAR_ALIAS_NAMESPACE: AtomicUsize = AtomicUsize::new(0);

/// Compile scalar recurrences into immutable Symbolica expressions.
///
/// This compiler does no numerical sampling and keeps no sample-valued cache.
/// Its output can be lowered into Symbolica's instruction-major evaluator.
pub struct ScalarCompiler {
    r: Vec<Vec<Atom>>,
    cache: HashMap<ScalarRequest, [Atom; 4]>,
    geometry: HashMap<Vec<usize>, Geometry>,
    exceptional_geometry: HashMap<Vec<usize>, ExceptionalGeometry>,
    regular_bubble_geometry: HashMap<Vec<usize>, Option<(Atom, Vec<Vec<Atom>>)>>,
    gram_regular_bubbles: bool,
    degenerate_families: HashMap<Vec<usize>, DegenerateFamily>,
    routing: Option<Vec<[Atom; 4]>>,
    routing_relations: HashMap<Vec<usize>, Option<RoutingRelation>>,
    lightlike_bubble: Option<LightlikeBubbleExpansion>,
    reduce_bubble_master: bool,
    reduce_degenerate_masters: bool,
    alias_enabled: bool,
    physical_poles: bool,
    six_dimensional_pentagon: bool,
    alias_namespace: usize,
    aliases: Vec<(Atom, Atom)>,
    pub leaves: Vec<MasterLeaf>,
}

impl ScalarCompiler {
    /// Construct from `R_ij = (p_i-p_j)^2 - m_i^2 - m_j^2`.
    pub fn new(r: Vec<Vec<Atom>>) -> Result<Self, String> {
        if r.is_empty() || r.iter().any(|row| row.len() != r.len()) {
            return Err("IBP requires a nonempty square Cayley R matrix".into());
        }
        Ok(Self {
            r,
            cache: HashMap::new(),
            geometry: HashMap::new(),
            exceptional_geometry: HashMap::new(),
            regular_bubble_geometry: HashMap::new(),
            gram_regular_bubbles: false,
            degenerate_families: HashMap::new(),
            routing: None,
            routing_relations: HashMap::new(),
            lightlike_bubble: None,
            reduce_bubble_master: false,
            reduce_degenerate_masters: false,
            alias_enabled: false,
            physical_poles: false,
            six_dimensional_pentagon: false,
            alias_namespace: SCALAR_ALIAS_NAMESPACE.fetch_add(1, Ordering::Relaxed),
            aliases: Vec::new(),
            leaves: Vec::new(),
        })
    }

    /// Compile shared Symbolica node handles instead of repeated inline trees.
    /// Register [`Self::aliases`] with the evaluator and use direct translation.
    pub fn with_aliases(r: Vec<Vec<Atom>>) -> Result<Self, String> {
        let mut compiler = Self::new(r)?;
        compiler.enable_aliases()?;
        Ok(compiler)
    }

    /// Enable DAG handles on a freshly constructed ordinary or constrained
    /// compiler. This must be called before compiling any request.
    pub fn enable_aliases(&mut self) -> Result<(), String> {
        if !self.cache.is_empty() {
            return Err("enable scalar aliases before compiling any requests".into());
        }
        self.alias_enabled = true;
        if let Some(expansion) = &mut self.lightlike_bubble {
            expansion.base.enable_aliases()?;
        }
        Ok(())
    }

    /// Shared exact definitions in dependency order. Laurent maps substitute
    /// node handles without expanding these definitions.
    pub fn aliases(&self) -> &[(Atom, Atom)] {
        &self.aliases
    }

    /// Permit genuine finite D6 pentagon leaves instead of an unsupported-master
    /// diagnostic. The numerical caller must enforce the qualified finite
    /// domain and propagate quadrature failures; this is not a box-only
    /// approximation or a claim about singular/massless endpoints.
    pub fn enable_six_dimensional_pentagon(&mut self) -> Result<(), String> {
        if !self.cache.is_empty() {
            return Err("enable the D6 pentagon backend before compiling requests".into());
        }
        self.six_dimensional_pentagon = true;
        Ok(())
    }

    /// Supply four-dimensional routing for exact denominator identities.
    /// The caller must ensure R's squared distances agree with these shifts;
    /// an independently supplied scalar-product matrix need not satisfy this.
    /// Relations are structural and cached by active propagator indices.
    pub fn set_routing(&mut self, shifts: Vec<[Atom; 4]>) -> Result<(), String> {
        self.set_certified_linear_coordinates(shifts)
    }

    /// Supply four linear coordinates on a certified external-momentum span.
    ///
    /// Unlike [`Self::set_routing`], these need not be Cartesian coordinates
    /// and their Minkowski squares need not reproduce R. The caller must
    /// certify, for every numerical sample and every Taylor component, that
    /// their linear dependencies exactly equal those of the external shifts.
    /// For example, four independent Gram covectors provide such coordinates
    /// on a certified rank-four span. Distances and denominator constants
    /// continue to come exclusively from the original R matrix.
    pub(crate) fn set_certified_linear_coordinates(
        &mut self,
        coordinates: Vec<[Atom; 4]>,
    ) -> Result<(), String> {
        if coordinates.len() != self.r.len() {
            return Err("scalar routing must have one four-vector per propagator".into());
        }
        if !self.cache.is_empty() {
            return Err("set scalar routing before compiling any requests".into());
        }
        self.routing = Some(coordinates);
        self.routing_relations.clear();
        Ok(())
    }

    /// Substitute guaranteed physical pole coefficients in master results:
    /// A0 has UV residue m² and no IR poles, B0 has UV residue one, and C0/D0
    /// have no UV pole. Leaves retain four independent parameter symbols so
    /// evaluator/backend parameter layouts remain unchanged. Enable this
    /// before any request; formal recurrence tests may leave it disabled.
    pub fn enable_physical_poles(&mut self) -> Result<(), String> {
        if !self.cache.is_empty() {
            return Err("enable physical pole identities before compiling any requests".into());
        }
        self.physical_poles = true;
        if let Some(expansion) = &mut self.lightlike_bubble {
            expansion.base.enable_physical_poles()?;
        }
        Ok(())
    }

    /// Use the analytic S-matrix recurrence for two-point requests satisfying
    /// `h >= sum(powers)-2`. This includes ordinary tensor descendants and
    /// avoids a Cayley-R denominator at thresholds and pseudothresholds.
    /// A lazy Symbolica branch retains the existing route when det(S)=0.
    pub fn enable_gram_regular_bubbles(&mut self) -> Result<(), String> {
        if !self.cache.is_empty() {
            return Err("enable Gram-regular bubbles before compiling requests".into());
        }
        self.gram_regular_bubbles = true;
        Ok(())
    }

    /// Compile an exact invariant constraint profile, retaining all unconstrained
    /// masses and squared distances as symbolic expressions. This handles
    /// massless propagators and lightlike edges inside arbitrary pinched families.
    ///
    /// The caller must verify every selected squared mass and squared distance
    /// vanishes as a *complete* numeric/Dual value, not merely in its primal part.
    /// Otherwise transverse derivatives would be discarded. The profile is
    /// structural and may be cached by these masks and the reduction shape;
    /// it contains no sample-valued numerical cache.
    pub fn with_zero_invariants(
        r: Vec<Vec<Atom>>,
        zero_mass_mask: &[bool],
        zero_distance_edges: &[(usize, usize)],
    ) -> Result<Self, String> {
        let n = r.len();
        if n == 0 || r.iter().any(|row| row.len() != n) || zero_mass_mask.len() != n {
            return Err(
                "zero-invariant profile needs a square R matrix and one mass flag per propagator"
                    .into(),
            );
        }
        let mut zero_distances = vec![vec![false; n]; n];
        for &(i, j) in zero_distance_edges {
            if i >= n || j >= n || i == j {
                return Err(
                    "zero-distance constraints require distinct valid propagator indices".into(),
                );
            }
            zero_distances[i][j] = true;
            zero_distances[j][i] = true;
        }
        let original_masses: Vec<_> = (0..n).map(|i| -&r[i][i] / Atom::num(2)).collect();
        let masses: Vec<_> = original_masses
            .iter()
            .enumerate()
            .map(|(i, mass)| {
                if zero_mass_mask[i] {
                    Atom::new()
                } else {
                    mass.clone()
                }
            })
            .collect();
        let constrained = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let distance = if i == j || zero_distances[i][j] {
                            Atom::new()
                        } else {
                            &r[i][j] + &original_masses[i] + &original_masses[j]
                        };
                        (distance - &masses[i] - &masses[j]).together()
                    })
                    .collect()
            })
            .collect();
        let mut compiler = Self::new(constrained)?;
        compiler.reduce_degenerate_masters = true;
        Ok(compiler)
    }

    /// A massless triangle with two exactly lightlike external invariants.
    /// `offshell_edge` names the remaining, nonzero-primal squared distance.
    ///
    /// The caller must verify that all three squared masses and the other two
    /// squared distances vanish in *every* Dual component. Transverse jets at
    /// these infrared boundaries are nonanalytic and must receive a diagnostic
    /// instead of having their derivative components silently discarded.
    pub fn massless_onshell_triangle(
        r: Vec<Vec<Atom>>,
        offshell_edge: (usize, usize),
    ) -> Result<Self, String> {
        if r.len() != 3
            || r.iter().any(|row| row.len() != 3)
            || offshell_edge.0 >= 3
            || offshell_edge.1 >= 3
            || offshell_edge.0 == offshell_edge.1
        {
            return Err(
                "massless on-shell triangle needs three propagators and a distinct off-shell edge"
                    .into(),
            );
        }
        let zero_edges: Vec<_> = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .filter(|&(i, j)| (i, j) != offshell_edge && (j, i) != offshell_edge)
            .collect();
        Self::with_zero_invariants(r, &[true; 3], &zero_edges)
    }

    /// A fully scaleless invariant family. All squared masses and pairwise
    /// squared distances must vanish as complete numeric/Dual values.
    /// Nonzero transverse derivatives are outside this nonanalytic profile.
    pub fn scaleless(propagators: usize) -> Result<Self, String> {
        Self::new(vec![vec![Atom::new(); propagators]; propagators])
    }

    /// Construct the analytic jet of a bubble about a lightlike momentum.
    ///
    /// The caller must restrict this plan to samples whose squared momentum
    /// difference has zero primal part. `derivative_order` is the maximal
    /// *total* nilpotent degree of the numeric Dual algebra (zero for ordinary
    /// numbers). The polynomial then preserves derivatives transverse to the
    /// lightlike surface, rather than replacing their Dual values by zero.
    ///
    /// If `equal_masses` is true, the masses must have equal, nonzero primal
    /// squared mass. Their difference is also expanded, preserving transverse
    /// mass derivatives. Otherwise the two primal squared masses must differ,
    /// and the exact lightlike partial-fraction recurrence is used at each
    /// Taylor order. This constructor does not implement a small-but-nonzero
    /// momentum approximation or a massless boundary expansion.
    pub fn lightlike_bubble(
        r: Vec<Vec<Atom>>,
        derivative_order: u16,
        equal_masses: bool,
    ) -> Result<Self, String> {
        if r.len() != 2 || r.iter().any(|row| row.len() != 2) {
            return Err("lightlike-bubble jets require exactly two propagators".into());
        }
        let two = Atom::num(2);
        let distance = &r[0][1] - (&r[0][0] + &r[1][1]) / &two;
        let mass_difference = (&r[0][0] - &r[1][1]) / &two;
        if equal_masses && r[0][0].is_zero() {
            return Err("a scaleless bubble has no analytic transverse mass/momentum jet".into());
        }
        let mut constrained = r.clone();
        constrained[0][1] = (&r[0][0] + &r[1][1]) / &two;
        constrained[1][0] = constrained[0][1].clone();
        let mut base = Box::new(Self::new(constrained)?);
        // Every Taylor coefficient must be evaluated at s=0; retaining a B0
        // leaf would pass the original nilpotent s to the scalar backend and
        // count those derivatives again. Partial fractions leave only A0s.
        base.reduce_bubble_master = true;
        let mut compiler = Self::new(r)?;
        compiler.lightlike_bubble = Some(LightlikeBubbleExpansion {
            distance,
            mass_difference,
            derivative_order,
            equal_masses,
            base,
            copied_aliases: 0,
        });
        Ok(compiler)
    }

    pub fn node_count(&self) -> usize {
        self.cache.len()
            + self
                .lightlike_bubble
                .as_ref()
                .map_or(0, |expansion| expansion.base.node_count())
    }

    /// Compile a request, preserving the native separate UV and IR slots.
    pub fn compile(&mut self, request: ScalarRequest) -> Result<[Atom; 4], String> {
        if request.powers.len() != self.r.len() {
            return Err(format!(
                "IBP power count {} differs from propagator count {}",
                request.powers.len(),
                self.r.len()
            ));
        }
        if let Some(value) = self.cache.get(&request) {
            return Ok(value.clone());
        }
        let active: Vec<_> = request
            .powers
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (*p != 0).then_some(i))
            .collect();
        let mut value = self.compile_uncached(&request, &active)?;
        let total: i32 = request.powers.iter().map(|&power| i32::from(power)).sum();
        if self.gram_regular_bubbles
            && self.lightlike_bubble.is_none()
            && active.len() == 2
            && i32::from(request.dimension_shift) >= total - 2
            && (request.dimension_shift != 0 || total != 2)
        {
            value = self.compile_gram_regular_bubble(&request, &active, value)?;
        }
        let value = self.cache_value(value);
        self.cache.insert(request, value.clone());
        Ok(value)
    }

    fn cache_value(&mut self, values: [Atom; 4]) -> [Atom; 4] {
        values.map(|body| self.cache_atom(body))
    }

    fn cache_atom(&mut self, body: Atom) -> Atom {
        if !self.alias_enabled || matches!(body.as_view(), AtomView::Num(_) | AtomView::Var(_)) {
            return body;
        }
        let alias = Atom::var(symbolica::symbol!(format!(
            "iregi::ibp_{}_node_{}",
            self.alias_namespace,
            self.aliases.len()
        )));
        self.aliases.push((alias.clone(), body));
        alias
    }

    fn compile_uncached(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
    ) -> Result<[Atom; 4], String> {
        if active.is_empty() {
            return Ok(zero());
        }
        if active.len() == 2 && self.lightlike_bubble.is_some() {
            return self.compile_lightlike_bubble(request);
        }
        let h = i32::from(request.dimension_shift);
        let total: i32 = request.powers.iter().map(|&p| i32::from(p)).sum();
        match self.degenerate_family(active) {
            DegenerateFamily::Scaleless => {
                // The Feynman quadratic vanishes identically. Beta integration
                // combines the propagator powers into the total power, even
                // when their momentum shifts differ by lightlike vectors.
                // Only the logarithmic case retains cancelling UV/IR poles.
                if total == h + 2 {
                    let denominator = Integer::factorial((total - 1) as u32);
                    let residue =
                        Atom::num(if total % 2 == 0 { 1 } else { -1 }) / Atom::num(denominator);
                    return Ok([residue.clone(), -residue, Atom::new(), Atom::new()]);
                }
                return Ok(zero());
            }
            DegenerateFamily::Coincident(anchor, merged) => {
                // Identical rows mean the scalar Feynman quadratic depends
                // on x_anchor+x_merged only. The exact Beta identity combines
                // their powers without a kinematic denominator or matrix solve.
                let mut combined = request.clone();
                combined.powers[anchor] = combined.powers[anchor]
                    .checked_add(combined.powers[merged])
                    .ok_or("combined scalar propagator power overflow")?;
                combined.powers[merged] = 0;
                return self.compile(combined);
            }
            DegenerateFamily::Distinct => {}
        }
        let raised = active.iter().position(|&i| request.powers[i] > 1);
        if active.len() >= 6 && self.routing.is_some() {
            let relation = self.routing_relation(active)?.ok_or(
                "no exact four-dimensional denominator relation for six or more propagators",
            )?;
            return self.compile_routing_relation(request, active, &relation);
        }
        if active.len() == 5 && h == 1 && raised.is_none() && self.routing.is_some() {
            if let Some(relation) = self.routing_relation(active)? {
                return self.compile_routing_relation(request, active, &relation);
            }
            if self.six_dimensional_pentagon {
                return Ok(self.pentagon6_master(active));
            }
            return Err("IBP dimension recurrence has a zero epsilon-independent divisor for 5 propagators at shift 1; ordinary four-dimensional tensors may use PV completion, but a standalone shifted scalar is not determined by the available A0-D0 Laurent orders".into());
        }
        if h == 0
            && raised.is_none()
            && active.len() <= 4
            && !(self.reduce_bubble_master && active.len() == 2)
        {
            if self.reduce_degenerate_masters && active.len() >= 3 {
                // Constrained profiles expose exact singularities which a
                // generic opaque C0/D0 leaf could hide from the reduction.
                // Preserve the one-mass/on-shell C0 master when S is invertible;
                // its zero dimension divisor is precisely the independent IR
                // master case. Other exact degeneracies reduce by the same
                // scalar relations as the reference before backend dispatch.
                match self.geometry(active) {
                    Ok(geometry) if !geometry.sum_b.is_zero() => return Ok(self.master(active)),
                    Ok(_) => {}
                    Err(_) => match self.exceptional_geometry(active)? {
                        ExceptionalGeometry::InvertibleS(_) if active.len() == 3 => {
                            return Ok(self.master(active));
                        }
                        _ => return self.compile_exceptional(request, active),
                    },
                }
            } else {
                return Ok(self.master(active));
            }
        }

        let geometry = match self.geometry(active) {
            Ok(geometry) => geometry,
            Err(_) => return self.compile_exceptional(request, active),
        };

        if geometry.sum_b.is_zero() {
            // Eq. (23): det(R)!=0, det(S)=0. The pinch identity holds for
            // arbitrary powers and dimension without a Gram denominator.
            let mut result = zero();
            for (i, &original_i) in active.iter().enumerate() {
                if geometry.b[i].is_zero() {
                    continue;
                }
                let value = self.compile(decrement(request, original_i))?;
                accumulate(&mut result, &geometry.b[i], &value);
            }
            return Ok(result);
        }

        if active.len() == 5 && h == 1 && raised.is_none() && self.six_dimensional_pentagon {
            return Ok(self.pentagon6_master(active));
        }

        // Scalar N>=5 reduction at epsilon^0. For N=5 the omitted
        // dimension-shifted remainder is O(epsilon), as in original IREGI.
        if h == 0 && raised.is_none() {
            let mut result = zero();
            for (i, &original_i) in active.iter().enumerate() {
                let child = decrement(request, original_i);
                let value = self.compile(child)?;
                accumulate(&mut result, &geometry.b[i], &value);
            }
            return Ok(result);
        }

        if h < 0 {
            // Eq. (15), expressed using b=R^-1*1 and B=sum(b):
            // J_h(v) = B*(5+2h-|v|-2eps)*J_(h+1)(v)
            //          + sum_i b_i J_h(v-e_i).
            let mut up = request.clone();
            up.dimension_shift += 1;
            let up_value = self.compile(up)?;
            let mut result = scale(
                &multiply_dimension(&up_value, 5 + 2 * h - total),
                &geometry.sum_b,
            );
            for (i, &original_i) in active.iter().enumerate() {
                let value = self.compile(decrement(request, original_i))?;
                accumulate(&mut result, &geometry.b[i], &value);
            }
            return Ok(result);
        }

        if raised.is_none() {
            let divisor = 3 + 2 * h - total;
            if divisor == 0 {
                // A lower-dimensional external span can make a shifted
                // pentagon reducible even though generic independent R symbols
                // do not expose that constraint. Generic rank-four pentagons
                // still require their genuine higher-dimensional master.
                if active.len() == 5 && self.routing.is_some() {
                    if let Some(relation) = self.routing_relation(active)? {
                        return self.compile_routing_relation(request, active, &relation);
                    }
                }
                return Err(format!(
                    "IBP dimension recurrence has a zero epsilon-independent divisor for {} propagators at shift {}; ordinary four-dimensional tensors may use PV completion, but a standalone shifted scalar is not determined by the available A0-D0 Laurent orders",
                    active.len(),
                    h
                ));
            }
            // Eq. (18): [J_(h-1)(v)-sum b_i J_(h-1)(v-e_i)]
            //             / [B*(3+2h-|v|-2eps)].
            let mut lower = request.clone();
            lower.dimension_shift -= 1;
            let mut result = self.compile(lower.clone())?;
            for (i, &original_i) in active.iter().enumerate() {
                let value = self.compile(decrement(&lower, original_i))?;
                accumulate(&mut result, &(-&geometry.b[i]), &value);
            }
            return Ok(scale(
                &divide_dimension(&result, divisor),
                &(Atom::num(1) / &geometry.sum_b),
            ));
        }

        let local_l = raised.expect("unit powers handled above");
        let original_l = active[local_l];
        let w = decrement(request, original_l);
        let divisor = Atom::num(i64::from(w.powers[original_l]));
        if h == 0 {
            // Eq. (19). Every child has fewer total propagator powers.
            let base = self.compile(w.clone())?;
            let common = scale(&multiply_dimension(&base, 5 - total), &Atom::num(-1));
            let mut result = zero();
            for (i, &original_i) in active.iter().enumerate() {
                let mut term = common.clone();
                let lower_i = decrement(&w, original_i);
                for &original_j in active {
                    if w.powers[original_j] == 0 {
                        continue;
                    }
                    let mut child = lower_i.clone();
                    child.powers[original_j] += 1;
                    let value = self.compile(child)?;
                    accumulate(
                        &mut term,
                        &Atom::num(i64::from(w.powers[original_j])),
                        &value,
                    );
                }
                accumulate(&mut result, &geometry.inverse[local_l][i], &term);
            }
            return Ok(scale(&result, &(Atom::num(1) / divisor)));
        }

        if 4 + 2 * h - total == 0 {
            // In the substituted relation below, J_h(w) is multiplied by
            // epsilon. Compiling it would request an otherwise unnecessary
            // D=6 pentagon (or its higher-dimensional analogue), whose
            // finite part is not determined by the available master orders.
            // Use the original Eq. (21), which only needs h-1 children.
            let mut lower = w;
            lower.dimension_shift -= 1;
            let b_l_over_b = &geometry.b[local_l] / &geometry.sum_b;
            let mut result = scale(&self.compile(lower.clone())?, &(-&b_l_over_b));
            for (i, &original_i) in active.iter().enumerate() {
                let coefficient = &b_l_over_b * &geometry.b[i] - &geometry.inverse[local_l][i];
                let value = self.compile(decrement(&lower, original_i))?;
                accumulate(&mut result, &coefficient, &value);
            }
            return Ok(scale(&result, &(Atom::num(1) / divisor)));
        }

        // Eq. (21), with the dimension recurrence substituted to eliminate
        // divisions by B. This is equivalent to the reference on nonsingular
        // kinematics and remains finite when B=0 but R is invertible:
        // (v_l-1) J_h(v) = -b_l (4+2h-|v|-2eps) J_h(v-e_l)
        //                         -sum_i Rinv_li J_(h-1)(v-e_l-e_i).
        let base = self.compile(w.clone())?;
        let mut result = scale(
            &multiply_dimension(&base, 4 + 2 * h - total),
            &(-&geometry.b[local_l]),
        );
        let mut lower = w;
        lower.dimension_shift -= 1;
        for (i, &original_i) in active.iter().enumerate() {
            let value = self.compile(decrement(&lower, original_i))?;
            accumulate(&mut result, &(-&geometry.inverse[local_l][i]), &value);
        }
        Ok(scale(&result, &(Atom::num(1) / divisor)))
    }

    fn degenerate_family(&mut self, active: &[usize]) -> DegenerateFamily {
        if let Some(&kind) = self.degenerate_families.get(active) {
            return kind;
        }
        let mut kind = DegenerateFamily::Distinct;
        if active
            .iter()
            .all(|&i| active.iter().all(|&j| self.r[i][j].is_zero()))
        {
            kind = DegenerateFamily::Scaleless;
        } else {
            'pairs: for (a, &i) in active.iter().enumerate() {
                for &j in &active[a + 1..] {
                    if active
                        .iter()
                        .all(|&k| (&self.r[i][k] - &self.r[j][k]).together().is_zero())
                    {
                        kind = DegenerateFamily::Coincident(i, j);
                        break 'pairs;
                    }
                }
            }
        }
        self.degenerate_families.insert(active.to_vec(), kind);
        kind
    }

    fn routing_relation(&mut self, active: &[usize]) -> Result<Option<RoutingRelation>, String> {
        if let Some(relation) = self.routing_relations.get(active) {
            return Ok(relation.clone());
        }
        let shifts = self
            .routing
            .as_ref()
            .ok_or("four-dimensional routing is unavailable")?;
        let origin = active[0];
        let mut rows: Vec<Vec<Atom>> = (0..4)
            .map(|mu| {
                active[1..]
                    .iter()
                    .map(|&i| &shifts[i][mu] - &shifts[origin][mu])
                    .collect()
            })
            .collect();
        // f_i=(p_i-p_0)^2-m_i²+m_0²=R_i0-R_00.
        rows.push(
            active[1..]
                .iter()
                .map(|&i| &self.r[i][origin] - &self.r[origin][origin])
                .collect(),
        );
        let matrix = Matrix::from_nested_vec(rows.clone(), exact_field())
            .map_err(|e| format!("routing denominator matrix: {e:?}"))?;
        let rhs = Matrix::new_vec(
            vec![
                Atom::new(),
                Atom::new(),
                Atom::new(),
                Atom::new(),
                Atom::num(1),
            ],
            exact_field(),
        );
        let mut solution = matrix.solve_any(&rhs).ok().map(|value| (value, true));
        if solution.is_none() {
            // If the constant identity is unavailable, a homogeneous relation
            // still transfers powers until a denominator is pinched. Symbolica
            // solves each constrained nullspace system; no manual elimination.
            for fixed in 0..active.len() - 1 {
                let mut constrained = rows.clone();
                let mut constraint = vec![Atom::new(); active.len() - 1];
                constraint[fixed] = Atom::num(1);
                constrained.push(constraint);
                let matrix = Matrix::from_nested_vec(constrained, exact_field())
                    .map_err(|e| format!("routing nullspace matrix: {e:?}"))?;
                let rhs = Matrix::new_vec(
                    vec![
                        Atom::new(),
                        Atom::new(),
                        Atom::new(),
                        Atom::new(),
                        Atom::new(),
                        Atom::num(1),
                    ],
                    exact_field(),
                );
                if let Ok(value) = matrix.solve_any(&rhs) {
                    solution = Some((value, false));
                    break;
                }
            }
        }
        let relation = solution.map(|(solution, constant_one)| {
            let mut coefficients = vec![Atom::new()];
            coefficients.extend((0..active.len() - 1).map(|i| solution[(i as u32, 0)].clone()));
            coefficients[0] = -coefficients[1..]
                .iter()
                .fold(Atom::new(), |sum, value| sum + value);
            let coefficients = coefficients
                .into_iter()
                .map(|value| self.cache_atom(value.together()))
                .collect();
            RoutingRelation {
                coefficients,
                constant_one,
            }
        });
        self.routing_relations
            .insert(active.to_vec(), relation.clone());
        Ok(relation)
    }

    fn compile_routing_relation(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
        relation: &RoutingRelation,
    ) -> Result<[Atom; 4], String> {
        let mut result = zero();
        if relation.constant_one {
            for (i, &original) in active.iter().enumerate() {
                if !relation.coefficients[i].is_zero() {
                    let value = self.compile(decrement(request, original))?;
                    accumulate(&mut result, &relation.coefficients[i], &value);
                }
            }
        } else {
            let anchor = active
                .iter()
                .enumerate()
                .filter(|(i, _)| !relation.coefficients[*i].is_zero())
                .max_by_key(|(_, original)| request.powers[**original])
                .map(|(i, _)| i)
                .ok_or("zero homogeneous routing relation")?;
            let mut raised = request.clone();
            raised.powers[active[anchor]] = raised.powers[active[anchor]]
                .checked_add(1)
                .ok_or("routing power transfer overflow")?;
            for (i, &original) in active.iter().enumerate() {
                if i != anchor && !relation.coefficients[i].is_zero() {
                    let value = self.compile(decrement(&raised, original))?;
                    accumulate(
                        &mut result,
                        &(-&relation.coefficients[i] / &relation.coefficients[anchor]),
                        &value,
                    );
                }
            }
        }
        Ok(result)
    }

    fn master(&mut self, active: &[usize]) -> [Atom; 4] {
        if let Some(leaf) = self
            .leaves
            .iter()
            .find(|leaf| leaf.kind == MasterLeafKind::FourDimensional && leaf.active == active)
        {
            return self.master_poles(active, leaf.slots.clone());
        }
        let id = active
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("_");
        let slots = std::array::from_fn(|slot| {
            Atom::var(symbolica::symbol!(format!("iregi::master_{id}_{slot}")))
        });
        self.leaves.push(MasterLeaf {
            kind: MasterLeafKind::FourDimensional,
            active: active.to_vec(),
            slots: slots.clone(),
        });
        self.master_poles(active, slots)
    }

    fn pentagon6_master(&mut self, active: &[usize]) -> [Atom; 4] {
        let slots = if let Some(leaf) = self.leaves.iter().find(|leaf| {
            leaf.kind == MasterLeafKind::SixDimensionalPentagon && leaf.active == active
        }) {
            leaf.slots.clone()
        } else {
            let id = active
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("_");
            // Preserve the uniform four-symbol parameter layout, even though
            // the qualified finite backend supplies zero for all pole slots.
            let slots = std::array::from_fn(|slot| {
                Atom::var(symbolica::symbol!(format!("iregi::master_d6_{id}_{slot}")))
            });
            self.leaves.push(MasterLeaf {
                kind: MasterLeafKind::SixDimensionalPentagon,
                active: active.to_vec(),
                slots: slots.clone(),
            });
            slots
        };
        [Atom::new(), Atom::new(), Atom::new(), slots[3].clone()]
    }

    fn master_poles(&self, active: &[usize], mut slots: [Atom; 4]) -> [Atom; 4] {
        if self.physical_poles {
            match active.len() {
                1 => {
                    slots[0] = -&self.r[active[0]][active[0]] / Atom::num(2);
                    slots[1] = Atom::new();
                    slots[2] = Atom::new();
                }
                2 => slots[0] = Atom::num(1),
                3 | 4 => slots[0] = Atom::new(),
                _ => unreachable!("only A0-D0 master leaves are constructed"),
            }
        }
        slots
    }

    fn compile_lightlike_bubble(&mut self, request: &ScalarRequest) -> Result<[Atom; 4], String> {
        let mut expansion = self.lightlike_bubble.take().expect("checked by caller");
        let result = expansion.compile(request);
        self.aliases
            .extend_from_slice(&expansion.base.aliases[expansion.copied_aliases..]);
        expansion.copied_aliases = expansion.base.aliases.len();
        for leaf in &expansion.base.leaves {
            if !self
                .leaves
                .iter()
                .any(|known| known.kind == leaf.kind && known.active == leaf.active)
            {
                self.leaves.push(leaf.clone());
            }
        }
        self.lightlike_bubble = Some(expansion);
        result
    }

    fn compile_gram_regular_bubble(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
        ordinary: [Atom; 4],
    ) -> Result<[Atom; 4], String> {
        if !self.regular_bubble_geometry.contains_key(active) {
            let mut rows = vec![vec![Atom::num(1); 3]; 3];
            rows[0][0] = Atom::Zero;
            for (i, &original_i) in active.iter().enumerate() {
                for (j, &original_j) in active.iter().enumerate() {
                    rows[i + 1][j + 1] = self.r[original_i][original_j].clone();
                }
            }
            let matrix = Matrix::from_nested_vec(rows, exact_field()).map_err(|error| {
                format!("cannot construct Symbolica bubble S matrix: {error:?}")
            })?;
            let determinant = matrix.det().map_err(|error| {
                format!("cannot calculate Symbolica bubble S determinant: {error:?}")
            })?;
            let geometry = if determinant.is_zero() {
                None
            } else {
                let inverse = matrix.inv().map_err(|error| {
                    format!("cannot invert Symbolica bubble S matrix: {error:?}")
                })?;
                let values = (0..3)
                    .map(|i| (0..3).map(|j| inverse[(i, j)].clone()).collect())
                    .collect();
                Some((self.cache_atom(determinant), values))
            };
            // This is not an exceptional-geometry entry: R may be invertible.
            self.regular_bubble_geometry
                .insert(active.to_vec(), geometry);
        }
        let Some((determinant, inverse)) = self.regular_bubble_geometry[active].clone() else {
            return Ok(ordinary);
        };
        let value = self.compile_s_bubble_value(request, active, &inverse)?;
        Ok(std::array::from_fn(|pole| {
            Symbol::IF.call((&determinant, &value[pole], &ordinary[pole]))
        }))
    }

    fn compile_s_bubble_value(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
        inverse: &[Vec<Atom>],
    ) -> Result<[Atom; 4], String> {
        if active.iter().any(|&i| request.powers[i] > 1) {
            self.compile_unsummed_s(request, active, inverse)
        } else {
            let mut lower = request.clone();
            lower.dimension_shift = lower
                .dimension_shift
                .checked_sub(1)
                .ok_or("IBP dimension shift underflow")?;
            // Full Eq. (18), expressed through S^-1. Retaining S^-1[0,0]
            // is essential for derivatives transverse to det(R)=0.
            let mut value = scale(&self.compile(lower.clone())?, &(-&inverse[0][0]));
            for (i, &original_i) in active.iter().enumerate() {
                accumulate(
                    &mut value,
                    &(-&inverse[i + 1][0]),
                    &self.compile(decrement(&lower, original_i))?,
                );
            }
            let total: i32 = request.powers.iter().map(|&power| i32::from(power)).sum();
            Ok(divide_dimension(
                &value,
                3 + 2 * i32::from(request.dimension_shift) - total,
            ))
        }
    }

    /// Eqs. (10), (14), (19): S * (0, w_j J_h(w+e_j)) equals
    /// (-J_(h-1)(w), T_i). The unsummed row is valid throughout det(S)!=0;
    /// unlike the shortened singular-R dimension relation it drops no term.
    fn compile_unsummed_s(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
        inverse: &[Vec<Atom>],
    ) -> Result<[Atom; 4], String> {
        let h = i32::from(request.dimension_shift);
        let total: i32 = request.powers.iter().map(|&power| i32::from(power)).sum();
        let local_l = active
            .iter()
            .position(|&i| request.powers[i] > 1)
            .ok_or("unsummed S recurrence requires a raised propagator power")?;
        let original_l = active[local_l];
        let w = decrement(request, original_l);
        let row = &inverse[local_l + 1];
        let mut lower = w.clone();
        lower.dimension_shift = lower
            .dimension_shift
            .checked_sub(1)
            .ok_or("IBP dimension shift underflow")?;
        let mut result = zero();
        if !row[0].is_zero() {
            accumulate(&mut result, &(-&row[0]), &self.compile(lower)?);
        }
        let base = self.compile(w.clone())?;
        let common = scale(
            &multiply_dimension(&base, 5 + 2 * h - total),
            &Atom::num(-1),
        );
        for (i, &original_i) in active.iter().enumerate() {
            if row[i + 1].is_zero() {
                continue;
            }
            let mut term = common.clone();
            let lower_i = decrement(&w, original_i);
            for &original_j in active {
                let mut child = lower_i.clone();
                child.powers[original_j] = child.powers[original_j]
                    .checked_add(1)
                    .ok_or("IBP propagator power overflow")?;
                accumulate(
                    &mut term,
                    &Atom::num(i64::from(w.powers[original_j])),
                    &self.compile(child)?,
                );
            }
            accumulate(&mut result, &row[i + 1], &term);
        }
        Ok(scale(
            &result,
            &(Atom::num(1) / Atom::num(i64::from(w.powers[original_l]))),
        ))
    }

    fn compile_exceptional(
        &mut self,
        request: &ScalarRequest,
        active: &[usize],
    ) -> Result<[Atom; 4], String> {
        let h = i32::from(request.dimension_shift);
        let total: i32 = request.powers.iter().map(|&p| i32::from(p)).sum();
        match self.exceptional_geometry(active)? {
            ExceptionalGeometry::InvertibleS(inverse) => {
                let a = 3 + 2 * h - total;
                if a != 0 {
                    // Eq. (22): det(R)=0, det(S)!=0, so c=0.
                    let mut lower = request.clone();
                    lower.dimension_shift = lower
                        .dimension_shift
                        .checked_sub(1)
                        .ok_or("IBP dimension shift underflow")?;
                    let mut result = zero();
                    for (i, &original_i) in active.iter().enumerate() {
                        if inverse[i + 1][0].is_zero() {
                            continue;
                        }
                        let value = self.compile(decrement(&lower, original_i))?;
                        accumulate(&mut result, &(-&inverse[i + 1][0]), &value);
                    }
                    return Ok(divide_dimension(&result, a));
                }
                self.compile_unsummed_s(request, active, &inverse)
            }
            ExceptionalGeometry::NullVector(vector) => {
                let c = -&vector[0];
                let mut result = zero();
                if !c.is_zero() {
                    // Eq. (26), which pinches at least one propagator power.
                    for (i, &original_i) in active.iter().enumerate() {
                        if vector[i + 1].is_zero() {
                            continue;
                        }
                        let value = self.compile(decrement(request, original_i))?;
                        accumulate(&mut result, &(&vector[i + 1] / &c), &value);
                    }
                } else {
                    // Eq. (28). Transfer power to a maximal-power nonzero
                    // component until another propagator is pinched.
                    let local_j = active
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| !vector[j + 1].is_zero())
                        .max_by_key(|(_, i)| request.powers[**i])
                        .map(|(j, _)| j)
                        .ok_or(
                            "Symbolica returned a zero vector for a constrained nullspace solve",
                        )?;
                    let mut up = request.clone();
                    up.powers[active[local_j]] = up.powers[active[local_j]]
                        .checked_add(1)
                        .ok_or("IBP propagator power overflow")?;
                    for (i, &original_i) in active.iter().enumerate() {
                        if i == local_j || vector[i + 1].is_zero() {
                            continue;
                        }
                        let value = self.compile(decrement(&up, original_i))?;
                        accumulate(
                            &mut result,
                            &(-&vector[i + 1] / &vector[local_j + 1]),
                            &value,
                        );
                    }
                }
                Ok(result)
            }
        }
    }

    fn exceptional_geometry(&mut self, active: &[usize]) -> Result<ExceptionalGeometry, String> {
        if let Some(geometry) = self.exceptional_geometry.get(active) {
            return Ok(geometry.clone());
        }
        let n = active.len();
        let mut rows = vec![vec![Atom::num(1); n + 1]; n + 1];
        rows[0][0] = Atom::new();
        for (i, &original_i) in active.iter().enumerate() {
            for (j, &original_j) in active.iter().enumerate() {
                rows[i + 1][j + 1] = self.r[original_i][original_j].clone();
            }
        }
        let matrix = Matrix::from_nested_vec(rows.clone(), exact_field())
            .map_err(|e| format!("cannot construct Symbolica Cayley S matrix: {e:?}"))?;
        if let Ok(inverse) = matrix.inv() {
            let values = (0..=n)
                .map(|i| {
                    (0..=n)
                        .map(|j| inverse[(i as u32, j as u32)].clone())
                        .collect()
                })
                .collect();
            let geometry = ExceptionalGeometry::InvertibleS(values);
            self.exceptional_geometry
                .insert(active.to_vec(), geometry.clone());
            return Ok(geometry);
        }

        // Symbolica solves the nullspace problem. A unit constraint makes the
        // requested vector nonzero; trying component zero first favors Eq.26.
        for component in 0..=n {
            let mut constrained = rows.clone();
            let mut constraint = vec![Atom::new(); n + 1];
            constraint[component] = Atom::num(1);
            constrained.push(constraint);
            let mut rhs = vec![Atom::new(); n + 2];
            rhs[n + 1] = Atom::num(1);
            let matrix = Matrix::from_nested_vec(constrained, exact_field()).map_err(|e| {
                format!("cannot construct constrained Symbolica Cayley matrix: {e:?}")
            })?;
            if let Ok(solution) = matrix.solve_any(&Matrix::new_vec(rhs, exact_field())) {
                let geometry = ExceptionalGeometry::NullVector(solution.into_vec());
                self.exceptional_geometry
                    .insert(active.to_vec(), geometry.clone());
                return Ok(geometry);
            }
        }
        Err(format!(
            "no usable Cayley recurrence for propagators {active:?}"
        ))
    }

    fn geometry(&mut self, active: &[usize]) -> Result<Geometry, String> {
        if self.exceptional_geometry.contains_key(active) {
            return Err("singular Cayley R matrix".into());
        }
        if let Some(geometry) = self.geometry.get(active) {
            return Ok(geometry.clone());
        }
        let rows = active
            .iter()
            .map(|&i| active.iter().map(|&j| self.r[i][j].clone()).collect())
            .collect();
        let field = exact_field();
        let matrix = Matrix::from_nested_vec(rows, field)
            .map_err(|e| format!("cannot construct Symbolica Cayley matrix: {e:?}"))?;
        let inverse = matrix
            .inv()
            .map_err(|e| format!("singular Cayley matrix for propagators {active:?}: {e:?}"))?;
        let inverse: Vec<Vec<Atom>> = (0..active.len())
            .map(|i| {
                (0..active.len())
                    .map(|j| inverse[(i as u32, j as u32)].clone())
                    .collect()
            })
            .collect();
        let b: Vec<_> = inverse
            .iter()
            .map(|row| {
                row.iter()
                    .fold(Atom::new(), |sum, entry| sum + entry)
                    .together()
            })
            .collect();
        let sum_b = b
            .iter()
            .fold(Atom::new(), |sum, entry| sum + entry)
            .together();
        let geometry = Geometry { inverse, b, sum_b };
        self.geometry.insert(active.to_vec(), geometry.clone());
        Ok(geometry)
    }
}

fn exact_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: Some(Box::new(|view, output| {
            *output = view.together();
            true
        })),
    }
}

impl LightlikeBubbleExpansion {
    fn compile(&mut self, request: &ScalarRequest) -> Result<[Atom; 4], String> {
        let mut result = zero();
        for momentum_order in 0..=self.derivative_order {
            let remaining = if self.equal_masses {
                self.derivative_order - momentum_order
            } else {
                0
            };
            for mass_order in 0..=remaining {
                // Feynman-parameter differentiation gives exactly
                // d_s^a d_delta^b J_h(v1,v2)
                // = (v1)_a (v2)_(a+b) J_(h+a)(v1+a,v2+a+b).
                // At s=delta=0, the two powers merge into one denominator.
                let mut child = request.clone();
                child.dimension_shift = child
                    .dimension_shift
                    .checked_add(
                        i16::try_from(momentum_order)
                            .map_err(|_| "bubble jet dimension overflow")?,
                    )
                    .ok_or("bubble jet dimension overflow")?;
                if self.equal_masses {
                    let power = u32::from(request.powers[0])
                        + u32::from(request.powers[1])
                        + 2 * u32::from(momentum_order)
                        + u32::from(mass_order);
                    child.powers = vec![
                        u16::try_from(power).map_err(|_| "bubble jet power overflow")?,
                        0,
                    ];
                } else {
                    for power in &mut child.powers {
                        *power = power
                            .checked_add(momentum_order)
                            .ok_or("bubble jet power overflow")?;
                    }
                }
                let mut coefficient = rising_factorial(request.powers[0], momentum_order)
                    * rising_factorial(request.powers[1], momentum_order + mass_order)
                    / Atom::num(Integer::factorial(u32::from(momentum_order)))
                    / Atom::num(Integer::factorial(u32::from(mass_order)));
                // The zeroth Taylor monomial is one, including at a literal
                // zero invariant; asking Symbolica for 0^0 would be undefined.
                if momentum_order != 0 {
                    coefficient = coefficient * self.distance.pow(i64::from(momentum_order));
                }
                if mass_order != 0 {
                    coefficient = coefficient * self.mass_difference.pow(i64::from(mass_order));
                }
                if coefficient.is_zero() {
                    continue;
                }
                let value = self.base.compile(child)?;
                accumulate(&mut result, &coefficient, &value);
            }
        }
        Ok(result)
    }
}

fn rising_factorial(start: u16, count: u16) -> Atom {
    (0..count).fold(Atom::num(1), |product, offset| {
        product * Atom::num(i64::from(start) + i64::from(offset))
    })
}

fn decrement(request: &ScalarRequest, index: usize) -> ScalarRequest {
    let mut child = request.clone();
    child.powers[index] -= 1;
    child
}

pub(crate) fn zero() -> [Atom; 4] {
    std::array::from_fn(|_| Atom::new())
}

pub(crate) fn scale(value: &[Atom; 4], factor: &Atom) -> [Atom; 4] {
    std::array::from_fn(|slot| &value[slot] * factor)
}

pub(crate) fn accumulate(result: &mut [Atom; 4], factor: &Atom, value: &[Atom; 4]) {
    for slot in 0..4 {
        result[slot] = &result[slot] + factor * &value[slot];
    }
}

/// Apply `(a-2epsilon)` using a cached Symbolica Laurent expansion.
pub(crate) fn multiply_dimension(value: &[Atom; 4], a: i32) -> [Atom; 4] {
    DIMENSION_MAPS.apply(value, a, false)
}

/// Apply `(a-2epsilon)^-1` through the finite term; `a` must be nonzero.
pub(crate) fn divide_dimension(value: &[Atom; 4], a: i32) -> [Atom; 4] {
    debug_assert_ne!(a, 0);
    DIMENSION_MAPS.apply(value, a, true)
}

struct DimensionMaps {
    a: Atom,
    values: [Atom; 4],
    multiply: [Atom; 4],
    divide: [Atom; 4],
}

static DIMENSION_MAPS: LazyLock<DimensionMaps> = LazyLock::new(|| {
    let eps_symbol = symbolica::symbol!("iregi_internal::laurent_eps");
    let eps = Atom::var(eps_symbol);
    let a = Atom::var(symbolica::symbol!("iregi_internal::laurent_a"));
    let values: [Atom; 4] = std::array::from_fn(|i| {
        Atom::var(symbolica::symbol!(format!(
            "iregi_internal::laurent_value{i}"
        )))
    });
    let uv = &values[0] / &eps;
    let ir = &values[1] / &eps + &values[2] / eps.pow(2);
    let total = &uv + &ir + &values[3];
    let factor = &a - Atom::num(2) * &eps;
    let derive = |factor: Atom| {
        // Symbolica owns Laurent multiplication, division and truncation.
        // Splitting the sources retains IREGI's physically distinct UV/IR
        // pole bookkeeping even when their single-pole residues cancel.
        let uv_series = (&factor * &uv)
            .series(eps_symbol, 0, 0)
            .expect("the universal UV dimension factor has a Laurent expansion");
        let ir_series = (&factor * &ir)
            .series(eps_symbol, 0, 0)
            .expect("the universal IR dimension factor has a Laurent expansion");
        let total_series = (&factor * &total)
            .series(eps_symbol, 0, 0)
            .expect("the universal dimension factor has a Laurent expansion");
        [
            uv_series
                .coefficient((-1).into())
                .expect("UV pole is within series precision"),
            ir_series
                .coefficient((-1).into())
                .expect("IR single pole is within series precision"),
            ir_series
                .coefficient((-2).into())
                .expect("IR double pole is within series precision"),
            total_series
                .coefficient(0.into())
                .expect("finite term is within series precision"),
        ]
    };
    let multiply = derive(factor.clone());
    let divide = derive(Atom::num(1) / factor);
    DimensionMaps {
        a,
        values,
        multiply,
        divide,
    }
});

impl DimensionMaps {
    fn apply(&self, values: &[Atom; 4], a: i32, divide: bool) -> [Atom; 4] {
        let replacements: Vec<_> = self
            .values
            .iter()
            .zip(values)
            .map(|(symbol, value)| Replacement::new(symbol.clone(), value.clone()))
            .chain([Replacement::new(self.a.clone(), Atom::num(a))])
            .collect();
        let map = if divide { &self.divide } else { &self.multiply };
        std::array::from_fn(|slot| map[slot].replace_multiple(replacements.iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::id::Replacement;

    fn equal(actual: &Atom, expected: Atom) {
        assert!(
            (actual - &expected).together().is_zero(),
            "actual {actual}, expected {expected}"
        );
    }

    fn expand_aliases(value: &Atom, aliases: &[(Atom, Atom)]) -> Atom {
        aliases
            .iter()
            .rev()
            .fold(value.clone(), |value, (alias, body)| {
                value.replace(alias.clone()).with(body.clone())
            })
    }

    #[test]
    fn analytic_bubble_s_rows_equal_ordinary_ibp_and_singular_limits() {
        let m0 = Atom::var(symbolica::symbol!("s_bubble_test::m0"));
        let m1 = Atom::var(symbolica::symbol!("s_bubble_test::m1"));
        let s = Atom::var(symbolica::symbol!("s_bubble_test::s"));
        let r = vec![
            vec![Atom::num(-2) * &m0, &s - &m0 - &m1],
            vec![&s - &m0 - &m1, Atom::num(-2) * &m1],
        ];
        let matrix = Matrix::from_nested_vec(
            vec![
                vec![Atom::Zero, Atom::num(1), Atom::num(1)],
                vec![Atom::num(1), r[0][0].clone(), r[0][1].clone()],
                vec![Atom::num(1), r[1][0].clone(), r[1][1].clone()],
            ],
            exact_field(),
        )
        .unwrap()
        .inv()
        .unwrap();
        let inverse: Vec<Vec<Atom>> = (0..3)
            .map(|i| (0..3).map(|j| matrix[(i, j)].clone()).collect())
            .collect();
        // At det(R)=0 the scalar master maps are not independent. Use their
        // analytic values at mu²=1: the B0 Feynman polynomial is (x+2)^2.
        let log2 = Symbol::LOG.call((Atom::num(2),));
        let log3 = Symbol::LOG.call((Atom::num(3),));
        let masters = [
            (
                "0",
                [
                    Atom::num(4),
                    Atom::Zero,
                    Atom::Zero,
                    Atom::num(4) * (Atom::num(1) - Atom::num(2) * &log2),
                ],
            ),
            (
                "1",
                [
                    Atom::num(9),
                    Atom::Zero,
                    Atom::Zero,
                    Atom::num(9) * (Atom::num(1) - Atom::num(2) * &log3),
                ],
            ),
            (
                "0_1",
                [
                    Atom::num(1),
                    Atom::Zero,
                    Atom::Zero,
                    Atom::num(2) + Atom::num(4) * &log2 - Atom::num(6) * &log3,
                ],
            ),
        ];
        let physical_masters = |mut expression: Atom| {
            for (id, values) in &masters {
                for (pole, value) in values.iter().enumerate() {
                    expression = expression
                        .replace(Atom::var(symbolica::symbol!(format!(
                            "iregi::master_{id}_{pole}"
                        ))))
                        .with(value.clone());
                }
            }
            expression
        };
        for (powers, dimension_shift) in [
            (vec![1, 1], 1),
            (vec![1, 1], 2),
            (vec![2, 1], 1),
            (vec![1, 2], 1),
            (vec![3, 1], 2),
            (vec![2, 2], 2),
        ] {
            let request = ScalarRequest {
                powers,
                dimension_shift,
            };
            let mut ordinary = ScalarCompiler::new(r.clone()).unwrap();
            let expected = ordinary.compile(request.clone()).unwrap();
            let mut analytic = ScalarCompiler::new(r.clone()).unwrap();
            let actual = analytic
                .compile_s_bubble_value(&request, &[0, 1], &inverse)
                .unwrap();
            for pole in 0..4 {
                equal(&actual[pole], expected[pole].clone());
            }
            // At the pseudothreshold, the original structurally singular-R
            // recurrence independently supplies the same Laurent values after
            // imposing the physical relation between the scalar masters.
            let mut singular = ScalarCompiler::new(vec![
                vec![Atom::num(-8), Atom::num(-12)],
                vec![Atom::num(-12), Atom::num(-18)],
            ])
            .unwrap();
            let expected = singular.compile(request).unwrap();
            for pole in 0..4 {
                let actual = actual[pole]
                    .together()
                    .replace(m0.clone())
                    .with(Atom::num(4))
                    .replace(m1.clone())
                    .with(Atom::num(9))
                    .replace(s.clone())
                    .with(Atom::num(1));
                equal(
                    &physical_masters(actual),
                    physical_masters(expected[pole].clone()),
                );
            }
        }
    }

    #[test]
    fn scalar_dag_preserves_inline_results_and_reuses_nodes() {
        let r = vec![
            vec![Atom::num(-4), Atom::num(-8), Atom::num(-9)],
            vec![Atom::num(-8), Atom::num(-6), Atom::num(-10)],
            vec![Atom::num(-9), Atom::num(-10), Atom::num(-8)],
        ];
        let mut inline = ScalarCompiler::new(r.clone()).unwrap();
        let mut dag = ScalarCompiler::with_aliases(r).unwrap();
        let request = ScalarRequest {
            powers: vec![3, 1, 1],
            dimension_shift: 2,
        };
        let expected = inline.compile(request.clone()).unwrap();
        let actual = dag.compile(request.clone()).unwrap();
        assert!(!dag.aliases().is_empty());
        for slot in 0..4 {
            equal(
                &expand_aliases(&actual[slot], dag.aliases()),
                expected[slot].clone(),
            );
        }
        let alias_count = dag.aliases().len();
        let node_count = dag.node_count();
        assert_eq!(dag.compile(request).unwrap(), actual);
        assert_eq!(dag.aliases().len(), alias_count);
        assert_eq!(dag.node_count(), node_count);
        assert!(dag.enable_aliases().is_err());
    }

    #[test]
    fn physical_poles_simplify_results_without_changing_leaf_parameters() {
        let mut compiler = ScalarCompiler::with_aliases(vec![vec![Atom::num(-6)]]).unwrap();
        compiler.enable_physical_poles().unwrap();
        let squared = compiler
            .compile(ScalarRequest {
                powers: vec![2],
                dimension_shift: 0,
            })
            .unwrap();
        equal(
            &expand_aliases(&squared[0], compiler.aliases()),
            Atom::num(1),
        );
        assert!(squared[1].is_zero());
        assert!(squared[2].is_zero());
        equal(
            &expand_aliases(&squared[3], compiler.aliases()),
            &compiler.leaves[0].slots[3] / Atom::num(3) - Atom::num(1),
        );
        assert!(
            compiler.leaves[0]
                .slots
                .iter()
                .all(|slot| matches!(slot.as_view(), AtomView::Var(_)))
        );
        assert!(compiler.enable_physical_poles().is_err());
    }

    fn routing_fixture(n: usize, homogeneous: bool) -> (Vec<[Atom; 4]>, Vec<Vec<Atom>>) {
        let points = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 1, 0, 1],
        ];
        let shifts: Vec<[Atom; 4]> = points[..n]
            .iter()
            .map(|point| point.map(Atom::num))
            .collect();
        let square = |p: &[Atom; 4]| p[0].pow(2) - p[1].pow(2) - p[2].pow(2) - p[3].pow(2);
        let masses: Vec<_> = shifts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if homogeneous {
                    square(p) + Atom::num(7)
                } else {
                    Atom::num(((i + 2) * (i + 2)) as u64)
                }
            })
            .collect();
        let r = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let delta = std::array::from_fn(|mu| &shifts[i][mu] - &shifts[j][mu]);
                        square(&delta) - &masses[i] - &masses[j]
                    })
                    .collect()
            })
            .collect();
        (shifts, r)
    }

    #[test]
    fn routing_relations_satisfy_exact_denominator_identities() {
        for (n, homogeneous) in [(6, false), (7, false), (6, true)] {
            let (shifts, r) = routing_fixture(n, homogeneous);
            let mut compiler = ScalarCompiler::new(r.clone()).unwrap();
            compiler.set_routing(shifts.clone()).unwrap();
            let active: Vec<_> = (0..n).collect();
            let relation = compiler.routing_relation(&active).unwrap().unwrap();
            assert_eq!(relation.constant_one, !homogeneous);
            equal(
                &relation
                    .coefficients
                    .iter()
                    .fold(Atom::new(), |sum, value| sum + value),
                Atom::new(),
            );
            for mu in 0..4 {
                equal(
                    &relation
                        .coefficients
                        .iter()
                        .enumerate()
                        .fold(Atom::new(), |sum, (i, value)| sum + value * &shifts[i][mu]),
                    Atom::new(),
                );
            }
            let constant = relation
                .coefficients
                .iter()
                .enumerate()
                .skip(1)
                .fold(Atom::new(), |sum, (i, value)| {
                    sum + value * (&r[i][0] - &r[0][0])
                });
            equal(&constant, Atom::num(if homogeneous { 0 } else { 1 }));
            assert_eq!(compiler.routing_relations.len(), 1);
        }
    }

    #[test]
    fn certified_covectors_preserve_cartesian_denominator_relations() {
        for homogeneous in [false, true] {
            let (shifts, r) = routing_fixture(6, homogeneous);
            // An invertible linear map of the true coordinates. Its Euclidean
            // or Minkowski squares deliberately differ from physical R.
            let covectors = shifts
                .iter()
                .map(|p| {
                    [
                        &p[0] + Atom::num(2) * &p[1],
                        &p[1] + &p[2],
                        &p[2] + Atom::num(3) * &p[3],
                        Atom::num(2) * &p[3],
                    ]
                })
                .collect();
            let mut compiler = ScalarCompiler::new(r.clone()).unwrap();
            compiler
                .set_certified_linear_coordinates(covectors)
                .unwrap();
            let relation = compiler
                .routing_relation(&[0, 1, 2, 3, 4, 5])
                .unwrap()
                .unwrap();
            assert_eq!(relation.constant_one, !homogeneous);
            equal(
                &relation
                    .coefficients
                    .iter()
                    .fold(Atom::new(), |sum, c| sum + c),
                Atom::new(),
            );
            for component in 0..4 {
                equal(
                    &relation
                        .coefficients
                        .iter()
                        .zip(&shifts)
                        .fold(Atom::new(), |sum, (c, p)| sum + c * &p[component]),
                    Atom::new(),
                );
            }
            let constant = relation
                .coefficients
                .iter()
                .enumerate()
                .skip(1)
                .fold(Atom::new(), |sum, (i, c)| sum + c * (&r[i][0] - &r[0][0]));
            equal(&constant, Atom::num(if homogeneous { 0 } else { 1 }));
        }
    }

    #[test]
    fn routing_reduction_preserves_generalized_powers_and_negative_dimensions() {
        let (shifts, r) = routing_fixture(6, false);
        let mut routed = ScalarCompiler::new(r.clone()).unwrap();
        routed.set_routing(shifts).unwrap();
        routed.enable_physical_poles().unwrap();
        let mut cayley = ScalarCompiler::new(r).unwrap();
        cayley.enable_physical_poles().unwrap();
        for request in [
            ScalarRequest {
                powers: vec![1; 6],
                dimension_shift: -1,
            },
            ScalarRequest {
                powers: vec![2, 1, 1, 1, 1, 1],
                dimension_shift: 0,
            },
        ] {
            let expected = cayley.compile(request.clone()).unwrap();
            let actual = routed.compile(request).unwrap();
            for slot in 0..4 {
                equal(&actual[slot], expected[slot].clone());
            }
        }
        assert!(!routed.geometry.contains_key(&vec![0, 1, 2, 3, 4, 5]));
        assert!(routed.leaves.iter().all(|leaf| leaf.active.len() <= 4));
    }

    #[test]
    fn lightlike_dag_imports_base_definitions_in_dependency_order() {
        let s = Atom::var(symbolica::symbol!("ibp_test::alias_jet_s"));
        let delta = Atom::var(symbolica::symbol!("ibp_test::alias_jet_delta"));
        let cross = &s - Atom::num(6) - &delta;
        let r = vec![
            vec![Atom::num(-6), cross.clone()],
            vec![cross, Atom::num(-6) - Atom::num(2) * &delta],
        ];
        let mut inline = ScalarCompiler::lightlike_bubble(r.clone(), 2, true).unwrap();
        let mut dag = ScalarCompiler::lightlike_bubble(r, 2, true).unwrap();
        dag.enable_aliases().unwrap();
        for request in [
            ScalarRequest {
                powers: vec![1, 1],
                dimension_shift: 0,
            },
            ScalarRequest {
                powers: vec![2, 1],
                dimension_shift: 1,
            },
        ] {
            let expected = inline.compile(request.clone()).unwrap();
            let actual = dag.compile(request).unwrap();
            for slot in 0..4 {
                equal(
                    &expand_aliases(&actual[slot], dag.aliases()),
                    expected[slot].clone(),
                );
            }
        }
        let unique: std::collections::HashSet<_> = dag.aliases().iter().map(|(a, _)| a).collect();
        assert_eq!(unique.len(), dag.aliases().len());
    }

    #[test]
    fn literal_lightlike_jet_retains_zeroth_taylor_coefficient() {
        for (r, equal_masses) in [
            (vec![vec![Atom::num(-6); 2]; 2], true),
            (
                vec![
                    vec![Atom::num(-4), Atom::num(-7)],
                    vec![Atom::num(-7), Atom::num(-10)],
                ],
                false,
            ),
        ] {
            let mut exact = ScalarCompiler::new(r.clone()).unwrap();
            exact.reduce_bubble_master = true;
            exact.enable_physical_poles().unwrap();
            for order in [0, 2] {
                let mut jet =
                    ScalarCompiler::lightlike_bubble(r.clone(), order, equal_masses).unwrap();
                jet.enable_physical_poles().unwrap();
                for request in [
                    ScalarRequest {
                        powers: vec![1, 1],
                        dimension_shift: 0,
                    },
                    ScalarRequest {
                        powers: vec![1, 1],
                        dimension_shift: 1,
                    },
                    ScalarRequest {
                        powers: vec![1, 3],
                        dimension_shift: 2,
                    },
                ] {
                    let expected = exact.compile(request.clone()).unwrap();
                    let actual = jet.compile(request).unwrap();
                    for slot in 0..4 {
                        equal(&actual[slot], expected[slot].clone());
                    }
                }
            }
        }
    }

    #[test]
    fn scaleless_families_keep_uv_ir_poles_without_backend_requests() {
        let mut compiler = ScalarCompiler::scaleless(3).unwrap();
        for (powers, dimension_shift, residue) in [
            (vec![1, 1, 0], 0, Atom::num(1)),
            (vec![1, 1, 1], 1, Atom::num(-1) / Atom::num(2)),
            (vec![2, 1, 1], 2, Atom::num(1) / Atom::num(6)),
        ] {
            assert_eq!(
                compiler
                    .compile(ScalarRequest {
                        powers,
                        dimension_shift
                    })
                    .unwrap(),
                [residue.clone(), -residue, Atom::new(), Atom::new()]
            );
        }
        assert_eq!(
            compiler
                .compile(ScalarRequest {
                    powers: vec![1, 1, 1],
                    dimension_shift: 0
                })
                .unwrap(),
            zero()
        );
        assert!(compiler.leaves.is_empty());
    }

    #[test]
    fn coincident_unit_denominators_do_not_request_redundant_bubble() {
        let mut compiler = ScalarCompiler::new(vec![vec![Atom::num(-6); 2]; 2]).unwrap();
        let bubble = compiler
            .compile(ScalarRequest {
                powers: vec![1, 1],
                dimension_shift: 0,
            })
            .unwrap();
        let raised = compiler
            .compile(ScalarRequest {
                powers: vec![2, 0],
                dimension_shift: 0,
            })
            .unwrap();
        assert_eq!(bubble, raised);
        assert_eq!(compiler.leaves.len(), 1);
        assert_eq!(compiler.leaves[0].active, vec![0]);
    }

    #[test]
    fn massless_triangle_profile_preserves_the_free_invariant() {
        let masses: [Atom; 3] = std::array::from_fn(|i| {
            Atom::var(symbolica::symbol!(format!("ibp_test::profile_m{i}")))
        });
        let s = Atom::var(symbolica::symbol!("ibp_test::profile_s"));
        let r = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let distance = if (i, j) == (0, 2) || (i, j) == (2, 0) {
                            s.clone()
                        } else {
                            Atom::new()
                        };
                        distance - &masses[i] - &masses[j]
                    })
                    .collect()
            })
            .collect();
        let mut compiler = ScalarCompiler::massless_onshell_triangle(r, (0, 2)).unwrap();
        equal(&compiler.r[0][2], s.clone());
        compiler.enable_aliases().unwrap();
        // Cover every scalar coefficient generated by Davydychev's tensor
        // representation through rank seven, including all metric pairings.
        for rank in 0..=7 {
            for pairs in 0..=rank / 2 {
                let degree = rank - 2 * pairs;
                for a in 0..=degree {
                    for b in 0..=degree - a {
                        let value = compiler
                            .compile(ScalarRequest {
                                powers: vec![a + 1, b + 1, degree - a - b + 1],
                                dimension_shift: (rank - pairs) as i16,
                            })
                            .unwrap();
                        assert!(!value.iter().all(AtomCore::is_zero));
                    }
                }
            }
        }
        assert!(compiler.leaves.iter().all(|leaf| leaf.active.len() >= 2));
    }

    #[test]
    fn zero_invariant_masks_preserve_unconstrained_masses_and_distances() {
        let masses: [Atom; 3] =
            std::array::from_fn(|i| Atom::var(symbolica::symbol!(format!("ibp_test::mask_m{i}"))));
        let distances: [Atom; 3] =
            std::array::from_fn(|i| Atom::var(symbolica::symbol!(format!("ibp_test::mask_s{i}"))));
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let mut r = vec![vec![Atom::new(); 3]; 3];
        for i in 0..3 {
            r[i][i] = Atom::num(-2) * &masses[i];
        }
        for ((i, j), distance) in pairs.into_iter().zip(&distances) {
            r[i][j] = distance - &masses[i] - &masses[j];
            r[j][i] = r[i][j].clone();
        }
        let mut compiler =
            ScalarCompiler::with_zero_invariants(r.clone(), &[true, false, true], &[(1, 0)])
                .unwrap();
        assert!(compiler.r[0][0].is_zero());
        assert!(compiler.r[2][2].is_zero());
        equal(&compiler.r[1][1], Atom::num(-2) * &masses[1]);
        equal(&compiler.r[0][1], -&masses[1]);
        equal(&compiler.r[0][2], distances[1].clone());
        equal(&compiler.r[1][2], &distances[2] - &masses[1]);
        let scaleless_pinch = compiler
            .compile(ScalarRequest {
                powers: vec![2, 0, 0],
                dimension_shift: 0,
            })
            .unwrap();
        assert_eq!(
            scaleless_pinch,
            [Atom::num(1), Atom::num(-1), Atom::new(), Atom::new()]
        );
        assert!(ScalarCompiler::with_zero_invariants(r.clone(), &[false; 2], &[]).is_err());
        assert!(ScalarCompiler::with_zero_invariants(r, &[false; 3], &[(3, 0)]).is_err());
    }

    #[test]
    fn singular_profile_reduces_unit_triangle_by_exact_partial_fractions() {
        let masses = [1, 2, 3];
        let r = masses
            .iter()
            .map(|&mi| masses.iter().map(|&mj| Atom::num(-mi - mj)).collect())
            .collect();
        let mut compiler =
            ScalarCompiler::with_zero_invariants(r, &[false; 3], &[(0, 1), (0, 2), (1, 2)])
                .unwrap();
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![1, 1, 1],
                dimension_shift: 0,
            })
            .unwrap();
        assert!(compiler.leaves.iter().all(|leaf| leaf.active.len() <= 2));
        assert!(matches!(
            compiler.exceptional_geometry.get(&vec![0, 1, 2]),
            Some(ExceptionalGeometry::NullVector(_))
        ));
        let finite_a: [Atom; 3] = std::array::from_fn(|i| {
            Atom::var(symbolica::symbol!(format!("ibp_test::partial_a{i}")))
        });
        let mut replacements = Vec::new();
        for leaf in &compiler.leaves {
            let (uv, finite) = if leaf.active.len() == 1 {
                let i = leaf.active[0];
                (Atom::num(masses[i]), finite_a[i].clone())
            } else {
                let (i, j) = (leaf.active[0], leaf.active[1]);
                (
                    Atom::num(1),
                    (&finite_a[i] - &finite_a[j]) / Atom::num(masses[i] - masses[j]),
                )
            };
            replacements.extend([
                Replacement::new(leaf.slots[0].clone(), uv),
                Replacement::new(leaf.slots[1].clone(), Atom::new()),
                Replacement::new(leaf.slots[2].clone(), Atom::new()),
                Replacement::new(leaf.slots[3].clone(), finite),
            ]);
        }
        for slot in &result[..3] {
            equal(&slot.replace_multiple(replacements.iter()), Atom::new());
        }
        equal(
            &result[3].replace_multiple(replacements.iter()),
            (&finite_a[0] - Atom::num(2) * &finite_a[1] + &finite_a[2]) / Atom::num(2),
        );
    }

    #[test]
    fn massive_tadpole_squared_is_mass_derivative() {
        let m2 = Atom::var(symbolica::symbol!("ibp_test::m2"));
        let mut compiler = ScalarCompiler::new(vec![vec![Atom::num(-2) * &m2]]).unwrap();
        let a0 = compiler
            .compile(ScalarRequest {
                powers: vec![1],
                dimension_shift: 0,
            })
            .unwrap();
        let squared = compiler
            .compile(ScalarRequest {
                powers: vec![2],
                dimension_shift: 0,
            })
            .unwrap();
        equal(&squared[0], &a0[0] / &m2);
        equal(&squared[1], (&a0[1] - &a0[2]) / &m2);
        equal(&squared[2], &a0[2] / &m2);
        equal(&squared[3], (&a0[3] - &a0[0] - &a0[1]) / &m2);
        assert_eq!(compiler.leaves.len(), 1);
    }

    #[test]
    fn dimension_shift_keeps_rational_finite_term() {
        let mut compiler = ScalarCompiler::new(vec![vec![Atom::num(-6)]]).unwrap();
        let a0 = compiler
            .compile(ScalarRequest {
                powers: vec![1],
                dimension_shift: 0,
            })
            .unwrap();
        let shifted = compiler
            .compile(ScalarRequest {
                powers: vec![1],
                dimension_shift: 1,
            })
            .unwrap();
        equal(&shifted[0], Atom::num(-3) * &a0[0] / Atom::num(2));
        equal(
            &shifted[3],
            Atom::num(-3) * &a0[3] / Atom::num(2)
                - Atom::num(3) * (&a0[0] + &a0[1]) / Atom::num(4)
                - Atom::num(3) * &a0[2] / Atom::num(8),
        );
    }

    #[test]
    fn scaleless_logarithm_retains_cancelling_poles() {
        let mut compiler = ScalarCompiler::new(vec![vec![Atom::new()]]).unwrap();
        let value = compiler
            .compile(ScalarRequest {
                powers: vec![2],
                dimension_shift: 0,
            })
            .unwrap();
        assert_eq!(
            value,
            [Atom::num(1), Atom::num(-1), Atom::new(), Atom::new()]
        );
        let lower = compiler
            .compile(ScalarRequest {
                powers: vec![1],
                dimension_shift: -1,
            })
            .unwrap();
        assert_eq!(
            lower,
            [Atom::num(-1), Atom::num(1), Atom::new(), Atom::new()]
        );
    }

    #[test]
    fn lowering_tadpole_dimension_matches_raising_its_power() {
        let mut compiler = ScalarCompiler::new(vec![vec![Atom::num(-6)]]).unwrap();
        let squared = compiler
            .compile(ScalarRequest {
                powers: vec![2],
                dimension_shift: 0,
            })
            .unwrap();
        let lower = compiler
            .compile(ScalarRequest {
                powers: vec![1],
                dimension_shift: -1,
            })
            .unwrap();
        for slot in 0..4 {
            equal(&lower[slot], -&squared[slot]);
        }
    }

    #[test]
    fn pinch_preserves_original_master_indices_and_cache() {
        let mut compiler = ScalarCompiler::new(vec![
            vec![Atom::num(-2), Atom::num(3)],
            vec![Atom::num(3), Atom::num(-4)],
        ])
        .unwrap();
        let request = ScalarRequest {
            powers: vec![0, 1],
            dimension_shift: 0,
        };
        let first = compiler.compile(request.clone()).unwrap();
        let second = compiler.compile(request).unwrap();
        assert_eq!(first, second);
        assert_eq!(compiler.leaves[0].active, vec![1]);
        assert_eq!(compiler.leaves.len(), 1);
        assert_eq!(compiler.node_count(), 1);
    }

    #[test]
    fn coincident_scalar_denominators_merge_powers_exactly() {
        let mut compiler = ScalarCompiler::new(vec![
            vec![Atom::num(-6), Atom::num(-6)],
            vec![Atom::num(-6), Atom::num(-6)],
        ])
        .unwrap();
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![2, 1],
                dimension_shift: 0,
            })
            .unwrap();
        // Both denominators coincide at invariant level, so the integral is
        // the third power of a single denominator with m^2=3.
        let leaf = compiler.leaves[0].slots.clone();
        let expected = scale(
            &multiply_dimension(&multiply_dimension(&leaf, 2), 0),
            &(Atom::num(1) / Atom::num(72)),
        );
        for slot in 0..4 {
            equal(&result[slot], expected[slot].clone());
        }
    }

    #[test]
    fn on_shell_massless_triangle_uses_invertible_s_branch() {
        let mut compiler = ScalarCompiler::new(vec![
            vec![Atom::new(), Atom::new(), Atom::num(-5)],
            vec![Atom::new(), Atom::new(), Atom::new()],
            vec![Atom::num(-5), Atom::new(), Atom::new()],
        ])
        .unwrap();
        // A raised-power tensor coefficient must not require R^-1 here.
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![2, 1, 1],
                dimension_shift: 1,
            })
            .unwrap();
        assert!(!result.iter().all(AtomCore::is_zero));
        assert!(compiler.exceptional_geometry.contains_key(&vec![0, 1, 2]));
    }

    #[test]
    fn pentagon_raised_power_avoids_unneeded_six_dimensional_master() {
        let r: Vec<Vec<Atom>> = (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| Atom::num(if i == j { -2 } else { 0 }))
                    .collect()
            })
            .collect();
        let mut compiler = ScalarCompiler::new(r).unwrap();
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![2, 1, 1, 1, 1],
                dimension_shift: 1,
            })
            .unwrap();
        assert!(!result.iter().all(AtomCore::is_zero));
        assert!(compiler.leaves.iter().all(|leaf| leaf.active.len() <= 4));
        let box_without_zero = compiler
            .leaves
            .iter()
            .find(|leaf| leaf.active == vec![1, 2, 3, 4])
            .unwrap();
        for slot in 0..4 {
            equal(&result[slot], &box_without_zero.slots[slot] / Atom::num(2));
        }
        assert!(!compiler.cache.contains_key(&ScalarRequest {
            powers: vec![1; 5],
            dimension_shift: 1,
        }));
    }

    #[test]
    fn genuine_pentagon_leaf_requires_opt_in_and_preserves_finite_slot() {
        let r: Vec<Vec<Atom>> = (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| Atom::num(if i == j { -2 } else { 0 }))
                    .collect()
            })
            .collect();
        let request = ScalarRequest {
            powers: vec![1; 5],
            dimension_shift: 1,
        };
        let mut disabled = ScalarCompiler::new(r.clone()).unwrap();
        assert!(disabled.compile(request.clone()).is_err());
        let mut compiler = ScalarCompiler::new(r).unwrap();
        compiler.enable_six_dimensional_pentagon().unwrap();
        let first = compiler.compile(request.clone()).unwrap();
        let second = compiler.compile(request).unwrap();
        assert_eq!(first, second);
        assert_eq!(compiler.leaves.len(), 1);
        assert_eq!(
            compiler.leaves[0].kind,
            MasterLeafKind::SixDimensionalPentagon
        );
        assert_eq!(compiler.leaves[0].active, vec![0, 1, 2, 3, 4]);
        assert!(first[..3].iter().all(AtomCore::is_zero));
        assert_eq!(first[3], compiler.leaves[0].slots[3]);
        assert!(compiler.enable_six_dimensional_pentagon().is_err());
    }

    #[test]
    fn equal_mass_lightlike_jet_has_transverse_and_mixed_derivatives() {
        let s = Atom::var(symbolica::symbol!("ibp_test::jet_s"));
        let delta = Atom::var(symbolica::symbol!("ibp_test::jet_delta"));
        let finite_a = Atom::var(symbolica::symbol!("ibp_test::jet_a_finite"));
        let cross = &s - Atom::num(6) - &delta;
        let r = vec![
            vec![Atom::num(-6), cross.clone()],
            vec![cross, Atom::num(-6) - Atom::num(2) * &delta],
        ];
        let mut compiler = ScalarCompiler::lightlike_bubble(r, 2, true).unwrap();
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![1, 1],
                dimension_shift: 0,
            })
            .unwrap();
        assert_eq!(compiler.leaves.len(), 1);
        assert_eq!(compiler.leaves[0].active, vec![0]);
        let slots = &compiler.leaves[0].slots;
        let substitutions = [
            Replacement::new(slots[0].clone(), Atom::num(3)),
            Replacement::new(slots[1].clone(), Atom::new()),
            Replacement::new(slots[2].clone(), Atom::new()),
            Replacement::new(slots[3].clone(), finite_a.clone()),
        ];
        // Independent expansion of -integral_0^1 log(3+delta*x-s*x*(1-x))dx.
        let expected = &finite_a / Atom::num(3) - Atom::num(1) + &s / Atom::num(18)
            - &delta / Atom::num(6)
            + s.pow(2) / Atom::num(540)
            - &s * &delta / Atom::num(108)
            + delta.pow(2) / Atom::num(54);
        equal(&result[3].replace_multiple(substitutions.iter()), expected);
        equal(
            &result[0].replace_multiple(substitutions.iter()),
            Atom::num(1),
        );
    }

    #[test]
    fn unequal_mass_lightlike_jet_uses_only_tadpoles() {
        let s = Atom::var(symbolica::symbol!("ibp_test::unequal_jet_s"));
        let finite_a = Atom::var(symbolica::symbol!("ibp_test::unequal_a_finite"));
        let finite_b = Atom::var(symbolica::symbol!("ibp_test::unequal_b_finite"));
        let r = vec![
            vec![Atom::num(-4), &s - Atom::num(7)],
            vec![&s - Atom::num(7), Atom::num(-10)],
        ];
        let mut compiler = ScalarCompiler::lightlike_bubble(r, 1, false).unwrap();
        let result = compiler
            .compile(ScalarRequest {
                powers: vec![1, 1],
                dimension_shift: 0,
            })
            .unwrap();
        assert!(compiler.leaves.iter().all(|leaf| leaf.active.len() == 1));
        let mut substitutions = Vec::new();
        for leaf in &compiler.leaves {
            let (mass, finite) = if leaf.active == vec![0] {
                (2, finite_a.clone())
            } else {
                (5, finite_b.clone())
            };
            substitutions.extend([
                Replacement::new(leaf.slots[0].clone(), Atom::num(mass)),
                Replacement::new(leaf.slots[1].clone(), Atom::new()),
                Replacement::new(leaf.slots[2].clone(), Atom::new()),
                Replacement::new(leaf.slots[3].clone(), finite),
            ]);
        }
        // Integral x(1-x)/(2+3x) dx = 7/18 - 10*log(5/2)/27.
        let expected = (&finite_b - &finite_a) / Atom::num(3)
            + &s * (Atom::num(7) / Atom::num(18) - Atom::num(5) * &finite_a / Atom::num(27)
                + Atom::num(2) * &finite_b / Atom::num(27));
        equal(&result[3].replace_multiple(substitutions.iter()), expected);
        equal(
            &result[0].replace_multiple(substitutions.iter()),
            Atom::num(1),
        );
    }
}
