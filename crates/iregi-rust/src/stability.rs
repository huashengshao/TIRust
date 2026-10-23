//! Precision-aware cancellation diagnostics for the compiled recurrence graph.
//!
//! These are conservative floating-point *estimates*, not certified error
//! bounds. Symbolica expands only individual, bounded recurrence bodies and
//! evaluates local monitors alongside the ordinary batch. A separate primal
//! batch propagates absolute scalar-master contributions through the shared
//! alias DAG; ABS is never differentiated to invent Taylor error bounds.
//! Aliases are not expanded recursively, and lazy branches keep analytic
//! exceptional-kinematics plans on the denominators they actually use.

pub(crate) mod arguments;
mod context;
pub(crate) use context::GuardContext;
pub(crate) mod quadrature;

use std::{
    collections::{BTreeSet, HashMap},
    num::FpCategory,
};

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::{atom::AtomField, float::Complex, integer::Integer, rational::Rational},
};

use crate::{
    ibp::MasterLeafKind,
    plan::ReductionPlan,
    types::{InputLayout, ReductionError},
};

const MAX_LOCAL_TERMS: usize = 256;

#[derive(Clone, Debug)]
struct Check {
    value: usize,
    terms: Vec<usize>,
    dimension: i32,
    operations: usize,
    denominator: bool,
    label: String,
}

#[derive(Clone, Debug)]
struct PrimalCheck {
    value: usize,
    amplitude: usize,
    dimension: i32,
    label: String,
}

/// Immutable monitor expressions for one reduction graph.
///
/// The executor appends `expressions()` to its ordinary expression roots and
/// evaluates them using the same aliases, numeric domain, and Dualizer.
/// `primal_expressions()` use a separate batch without the Dualizer.
#[derive(Clone, Debug)]
pub(crate) struct StabilityPlan {
    expressions: Vec<Atom>,
    predicate_aliases: Vec<(Atom, Atom)>,
    master_activations: Vec<Atom>,
    master_activation_aliases: Vec<(Atom, Atom)>,
    primal_expressions: Vec<Atom>,
    primal_aliases: Vec<(Atom, Atom)>,
    primal_checks: Vec<PrimalCheck>,
    snapping_expressions: Vec<Atom>,
    snapping_aliases: Vec<(Atom, Atom)>,
    snapping_labels: Vec<String>,
    snapping_threshold: Option<Atom>,
    snapping_zero_image: Option<Atom>,
    checks: Vec<Check>,
    output_dimensions: Vec<i32>,
    layout: InputLayout,
}

impl StabilityPlan {
    pub(crate) fn prepare(plan: &ReductionPlan) -> Result<Self, ReductionError> {
        let mut monitor = Self {
            expressions: Vec::new(),
            predicate_aliases: Vec::new(),
            master_activations: Vec::new(),
            master_activation_aliases: Vec::new(),
            primal_expressions: Vec::new(),
            primal_aliases: Vec::new(),
            primal_checks: Vec::new(),
            snapping_expressions: Vec::new(),
            snapping_aliases: Vec::new(),
            snapping_labels: Vec::new(),
            snapping_threshold: None,
            snapping_zero_image: None,
            checks: Vec::new(),
            output_dimensions: Vec::new(),
            layout: plan.layout.clone(),
        };
        let mut indices = HashMap::new();
        let mut dimensions = HashMap::new();
        for (i, parameter) in plan.parameters.iter().enumerate() {
            let squared = plan.layout.masses.contains(&i)
                || plan
                    .layout
                    .scalar_products
                    .as_ref()
                    .is_some_and(|range| range.contains(&i));
            dimensions.insert(parameter.clone(), if squared { 2 } else { 1 });
        }
        for leaf in &plan.leaves {
            let integration_dimension = match leaf.kind {
                MasterLeafKind::FourDimensional => 4,
                MasterLeafKind::SixDimensionalPentagon => 6,
            };
            for slot in &leaf.slots {
                dimensions.insert(
                    slot.clone(),
                    integration_dimension - 2 * leaf.active.len() as i32,
                );
            }
        }
        let definitions: HashMap<_, _> = plan.aliases.iter().cloned().collect();
        // Propagate use conditions backwards through the topologically ordered
        // aliases. A union of paths is represented by Symbolica's lazy IF;
        // inactive generic branches must never become eager monitor roots.
        let mut uses: HashMap<Atom, BTreeSet<Atom>> = HashMap::new();
        let mut predicate_indices = HashMap::new();
        let always = Atom::num(1);
        for expression in plan.moments.iter().flatten() {
            collect_alias_uses(expression.as_view(), &always, &definitions, &mut uses);
        }
        let mut reachable = HashMap::new();
        for (alias, body) in plan.aliases.iter().rev() {
            if let Some(paths) = uses.remove(alias) {
                let active = monitor.intern_predicate(union_paths(paths), &mut predicate_indices);
                collect_alias_uses(body.as_view(), &active, &definitions, &mut uses);
                reachable.insert(alias.clone(), active);
            }
        }
        let mut denominators = BTreeSet::new();
        for (alias, body) in &plan.aliases {
            let dimension = mass_dimension(body.as_view(), &dimensions).unwrap_or(0);
            dimensions.insert(alias.clone(), dimension);
            let Some(active) = reachable.get(alias) else {
                continue;
            };
            monitor.add_branch_sums(
                body,
                active,
                dimension,
                false,
                format!("recurrence {alias}"),
                &dimensions,
                &mut indices,
            );
            collect_denominators(body.as_view(), active, &mut denominators);
        }
        let base_dimension = 4 + 2 * i32::from(plan.settings.dimension_shift)
            - 2 * plan
                .settings
                .powers
                .iter()
                .map(|&p| i32::from(p))
                .sum::<i32>();
        for (basis, slots) in plan.basis.iter().zip(&plan.moments) {
            let dimension = base_dimension + basis.rank() as i32;
            for (pole, expression) in slots.iter().enumerate() {
                monitor.output_dimensions.push(dimension);
                monitor.add_branch_sums(
                    expression,
                    &always,
                    dimension,
                    false,
                    format!("moment {:?}, Laurent slot {pole}", basis.0),
                    &dimensions,
                    &mut indices,
                );
                collect_denominators(expression.as_view(), &always, &mut denominators);
            }
        }
        for (denominator, active) in denominators {
            // A denominator may be a named invariant. Inspect that one body
            // as well, without substituting the entire upstream graph.
            let body = definitions.get(&denominator).unwrap_or(&denominator);
            let dimension = mass_dimension(body.as_view(), &dimensions).unwrap_or(0);
            monitor.add_branch_sums(
                body,
                &active,
                dimension,
                true,
                format!("denominator {denominator}"),
                &dimensions,
                &mut indices,
            );
        }
        monitor.prepare_primal_amplitudes(plan, &reachable, base_dimension)?;
        monitor.prepare_master_activations(plan);
        monitor.prepare_snapping_diagnostics(plan)?;
        Ok(monitor)
    }

    pub(crate) fn expressions(&self) -> &[Atom] {
        &self.expressions
    }

    pub(crate) fn aliases(&self) -> &[(Atom, Atom)] {
        &self.predicate_aliases
    }

    /// Master-contribution amplitudes are evaluated on primals only. Applying
    /// the Dualizer to ABS would not produce bounds for Taylor coefficients.
    pub(crate) fn primal_expressions(&self) -> &[Atom] {
        &self.primal_expressions
    }

    pub(crate) fn primal_aliases(&self) -> &[(Atom, Atom)] {
        &self.primal_aliases
    }

    /// Primal strict-comparison residuals, guarded by the demand for their
    /// owning predicates. Unused queries are omitted; inactive paths yield one.
    pub(crate) fn snapping_expressions(&self) -> &[Atom] {
        &self.snapping_expressions
    }

    pub(crate) fn snapping_aliases(&self) -> &[(Atom, Atom)] {
        &self.snapping_aliases
    }

    pub(crate) fn snapping_labels(&self) -> &[String] {
        &self.snapping_labels
    }

    /// A representability probe for the threshold, equal to one when no
    /// snapping predicate is demanded. Exact zero remains zero on active paths.
    pub(crate) fn snapping_threshold(&self) -> Option<&Atom> {
        self.snapping_threshold.as_ref()
    }

    /// Compatibility probe for compilers that require a zero Gram/Cayley
    /// diagonal. It is zero on paths without a relevant snapping operation.
    pub(crate) fn snapping_zero_image(&self) -> Option<&Atom> {
        self.snapping_zero_image.as_ref()
    }

    fn prepare_snapping_diagnostics(&mut self, plan: &ReductionPlan) -> Result<(), ReductionError> {
        if plan.snapping_diagnostics.is_empty() && !plan.snapping_zero_image_required {
            return Ok(());
        }
        // Activation aliases contain a subset of existing aliases plus fresh
        // predicates. Keep their topological order without duplicate names.
        let mut seen = BTreeSet::new();
        let aliases: Vec<_> = plan
            .aliases
            .iter()
            .chain(&self.predicate_aliases)
            .chain(&self.primal_aliases)
            .chain(&self.master_activation_aliases)
            .filter(|(alias, _)| seen.insert(alias.clone()))
            .cloned()
            .collect();
        let definitions: HashMap<_, _> = aliases.iter().cloned().collect();
        let masters: BTreeSet<_> = plan
            .leaves
            .iter()
            .flat_map(|leaf| leaf.slots.iter().cloned())
            .collect();
        let mut safe_aliases = BTreeSet::new();
        for (alias, body) in &aliases {
            if parameter_only(body, &definitions, &masters, &safe_aliases) {
                safe_aliases.insert(alias.clone());
            }
        }
        let always = Atom::num(1);
        let mut uses = HashMap::new();
        for root in plan
            .moments
            .iter()
            .flatten()
            .chain(&self.expressions)
            .chain(&self.primal_expressions)
        {
            collect_alias_uses(root.as_view(), &always, &definitions, &mut uses);
        }
        for (arguments, active) in plan.master_arguments.iter().zip(&self.master_activations) {
            // The activation predicate itself must be classified, but scalar
            // arguments are relevant only when this leaf is requested.
            collect_alias_uses(active.as_view(), &always, &definitions, &mut uses);
            for argument in arguments {
                collect_alias_uses(argument.as_view(), active, &definitions, &mut uses);
            }
        }
        // Match the executor's actual classifiers. General zero-profile
        // dispatch is disabled for legacy-snapped PV, whose scalar products
        // need not be bilinear; whole lightlike bubbles still use their own
        // analytic classifier. A discarded classifier must not demand probes.
        if plan.zero_profile_dispatch_allowed() {
            for i in 0..plan.settings.propagators {
                for j in i + 1..plan.settings.propagators {
                    collect_alias_uses(
                        plan.distances[i][j].as_view(),
                        &always,
                        &definitions,
                        &mut uses,
                    );
                }
            }
        } else if plan.settings.propagators == 2 && !plan.special_kinematics {
            collect_alias_uses(
                plan.distances[0][1].as_view(),
                &always,
                &definitions,
                &mut uses,
            );
        }
        let mut predicate_indices = HashMap::new();
        let mut predicates = Vec::new();
        let mut demanded = HashMap::new();
        for (alias, body) in aliases.iter().rev() {
            let Some(paths) = uses.remove(alias) else {
                continue;
            };
            let union = union_paths(paths);
            // Diagnostics run before masters. A master-conditioned use must
            // conservatively request the parameter-only comparison rather
            // than introduce a circular pre-master evaluator dependency.
            let active = if parameter_only(&union, &definitions, &masters, &safe_aliases) {
                intern_named_predicate(
                    union,
                    "iregi::snapping_active_path",
                    &mut predicates,
                    &mut predicate_indices,
                )
            } else {
                always.clone()
            };
            collect_alias_uses(body.as_view(), &active, &definitions, &mut uses);
            demanded.insert(alias.clone(), active);
        }
        let mut active_paths = BTreeSet::new();
        for diagnostic in &plan.snapping_diagnostics {
            let Some(active) = demanded.get(&diagnostic.owner) else {
                continue;
            };
            if !parameter_only(
                &diagnostic.expression,
                &definitions,
                &masters,
                &safe_aliases,
            ) {
                return Err(ReductionError::Preparation(format!(
                    "snapping diagnostic depends on a scalar master: {}",
                    diagnostic.label
                )));
            }
            self.snapping_expressions
                .push(guarded(active, diagnostic.expression.clone(), true));
            self.snapping_labels.push(diagnostic.label.clone());
            active_paths.insert(active.clone());
        }
        if self.snapping_expressions.is_empty() && !plan.snapping_zero_image_required {
            return Ok(());
        }
        // A diagonal-only IBP tadpole may require the compatibility probe even
        // though no ordinary off-diagonal query appears in its reduced graph.
        let any_active = if plan.snapping_zero_image_required {
            always
        } else {
            union_paths(active_paths)
        };
        self.snapping_threshold = plan
            .snapping_threshold
            .as_ref()
            .map(|threshold| guarded(&any_active, threshold.clone(), true));
        self.snapping_zero_image = plan
            .snapping_zero_image
            .as_ref()
            .map(|value| guarded(&any_active, value.clone(), false));
        let all_aliases: Vec<_> = aliases.into_iter().chain(predicates).collect();
        let all_definitions: HashMap<_, _> = all_aliases.iter().cloned().collect();
        let mut reachable = BTreeSet::new();
        let mut pending: Vec<_> = self
            .snapping_expressions
            .iter()
            .chain(self.snapping_threshold.iter())
            .chain(self.snapping_zero_image.iter())
            .cloned()
            .collect();
        while let Some(expression) = pending.pop() {
            for symbol in expression.get_all_symbols(false) {
                let alias = Atom::var(symbol);
                if let Some(body) = all_definitions.get(&alias) {
                    if reachable.insert(alias) {
                        pending.push(body.clone());
                    }
                }
            }
        }
        self.snapping_aliases = all_aliases
            .into_iter()
            .filter(|(alias, _)| reachable.contains(alias))
            .collect();
        Ok(())
    }

    fn prepare_primal_amplitudes(
        &mut self,
        plan: &ReductionPlan,
        reachable: &HashMap<Atom, Atom>,
        base_dimension: i32,
    ) -> Result<(), ReductionError> {
        let mut amplitudes: HashMap<_, _> = plan
            .leaves
            .iter()
            .flat_map(|leaf| leaf.slots.iter())
            .map(|slot| (slot.clone(), Symbol::ABS.call((slot,))))
            .collect();
        for (alias, body) in &plan.aliases {
            if !reachable.contains_key(alias) || !depends_on_amplitudes(body, &amplitudes) {
                continue;
            }
            let amplitude = master_amplitude(body, &amplitudes)?;
            let handle = Atom::var(symbolica::symbol!(format!(
                "iregi::master_amplitude_{}",
                self.primal_aliases.len()
            )));
            self.primal_aliases.push((handle.clone(), amplitude));
            amplitudes.insert(alias.clone(), handle);
        }
        for (basis, slots) in plan.basis.iter().zip(&plan.moments) {
            for (pole, expression) in slots.iter().enumerate() {
                if !depends_on_amplitudes(expression, &amplitudes) {
                    continue;
                }
                let amplitude = master_amplitude(expression, &amplitudes)?;
                let value = self.primal_expressions.len();
                self.primal_expressions.push(expression.clone());
                let amplitude_index = self.primal_expressions.len();
                self.primal_expressions.push(amplitude);
                self.primal_checks.push(PrimalCheck {
                    value,
                    amplitude: amplitude_index,
                    dimension: base_dimension + basis.rank() as i32,
                    label: format!("moment {:?}, Laurent slot {pole}", basis.0),
                });
            }
        }
        Ok(())
    }

    /// One kinematic predicate per scalar leaf, including monitor demand.
    /// Master-dependent branch conditions conservatively activate their users
    /// before scalar evaluation, avoiding circular pre-master dependencies.
    pub(crate) fn master_activations(&self) -> &[Atom] {
        &self.master_activations
    }

    pub(crate) fn master_activation_aliases(&self) -> &[(Atom, Atom)] {
        &self.master_activation_aliases
    }

    fn prepare_master_activations(&mut self, plan: &ReductionPlan) {
        let aliases: Vec<_> = plan
            .aliases
            .iter()
            .chain(&self.predicate_aliases)
            .chain(&self.primal_aliases)
            .cloned()
            .collect();
        let mut definitions: HashMap<_, _> = aliases.iter().cloned().collect();
        let masters: BTreeSet<_> = plan
            .leaves
            .iter()
            .flat_map(|leaf| leaf.slots.iter().cloned())
            .collect();
        let mut safe_aliases = BTreeSet::new();
        for (alias, body) in &aliases {
            if parameter_only(body, &definitions, &masters, &safe_aliases) {
                safe_aliases.insert(alias.clone());
            }
        }
        // Register leaves as traversal targets, not alias definitions to expand.
        for slot in &masters {
            definitions.insert(slot.clone(), Atom::new());
        }
        let mut uses = HashMap::new();
        let always = Atom::num(1);
        for expression in plan
            .moments
            .iter()
            .flatten()
            .chain(&self.expressions)
            .chain(&self.primal_expressions)
        {
            collect_alias_uses(expression.as_view(), &always, &definitions, &mut uses);
        }
        let mut predicates = Vec::new();
        let mut predicate_indices = HashMap::new();
        for (alias, body) in aliases.iter().rev() {
            if let Some(paths) = uses.remove(alias) {
                let union = union_paths(paths);
                let active = if parameter_only(&union, &definitions, &masters, &safe_aliases) {
                    intern_activation_predicate(union, &mut predicates, &mut predicate_indices)
                } else {
                    always.clone()
                };
                collect_alias_uses(body.as_view(), &active, &definitions, &mut uses);
            }
        }
        for leaf in &plan.leaves {
            let mut paths = BTreeSet::new();
            for slot in &leaf.slots {
                if let Some(slot_paths) = uses.remove(slot) {
                    paths.extend(slot_paths);
                }
            }
            let union = union_paths(paths);
            let active = if parameter_only(&union, &definitions, &masters, &safe_aliases) {
                intern_activation_predicate(union, &mut predicates, &mut predicate_indices)
            } else {
                always.clone()
            };
            self.master_activations.push(active);
        }
        // Keep only the parameter/predicate closure actually used by the
        // activation roots; never hand scalar-valued aliases to the pre-master
        // evaluator. Newly interned predicates are safe by construction.
        let all_aliases: Vec<_> = aliases.into_iter().chain(predicates).collect();
        let all_definitions: HashMap<_, _> = all_aliases.iter().cloned().collect();
        let mut reachable = BTreeSet::new();
        let mut pending = self.master_activations.clone();
        while let Some(expression) = pending.pop() {
            for symbol in expression.get_all_symbols(false) {
                let alias = Atom::var(symbol);
                if let Some(body) = all_definitions.get(&alias) {
                    if reachable.insert(alias) {
                        pending.push(body.clone());
                    }
                }
            }
        }
        self.master_activation_aliases = all_aliases
            .into_iter()
            .filter(|(alias, _)| reachable.contains(alias))
            .collect();
    }

    fn intern_predicate(&mut self, expression: Atom, indices: &mut HashMap<Atom, Atom>) -> Atom {
        if expression == Atom::num(1) {
            return expression;
        }
        if let Some(alias) = indices.get(&expression) {
            return alias.clone();
        }
        let alias = Atom::var(symbolica::symbol!(format!(
            "iregi::stability_path_{}",
            self.predicate_aliases.len()
        )));
        indices.insert(expression.clone(), alias.clone());
        self.predicate_aliases.push((alias.clone(), expression));
        alias
    }

    fn intern(&mut self, expression: Atom, indices: &mut HashMap<Atom, usize>) -> usize {
        if let Some(&index) = indices.get(&expression) {
            return index;
        }
        let index = self.expressions.len();
        indices.insert(expression.clone(), index);
        self.expressions.push(expression);
        index
    }

    fn add_branch_sums(
        &mut self,
        expression: &Atom,
        active: &Atom,
        dimension: i32,
        denominator: bool,
        label: String,
        dimensions: &HashMap<Atom, i32>,
        indices: &mut HashMap<Atom, usize>,
    ) {
        if let Some((condition, yes, no)) = if_parts(expression.as_view()) {
            self.add_branch_sums(
                &yes.to_owned(),
                &branch_path(active, condition, true),
                dimension,
                denominator,
                label.clone(),
                dimensions,
                indices,
            );
            self.add_branch_sums(
                &no.to_owned(),
                &branch_path(active, condition, false),
                dimension,
                denominator,
                label,
                dimensions,
                indices,
            );
            return;
        }
        self.add_sum(
            expression,
            active,
            dimension,
            denominator,
            label.clone(),
            indices,
        );
        // IF can occur below a product or sum, too. Inspect its selected branch
        // without treating an inactive child's denominator as an eager root.
        visit_if_branches(expression.as_view(), active, &mut |branch, path| {
            let child_dimension = mass_dimension(branch, dimensions).unwrap_or(dimension);
            self.add_branch_sums(
                &branch.to_owned(),
                path,
                child_dimension,
                false,
                label.clone(),
                dimensions,
                indices,
            );
        });
    }

    fn add_sum(
        &mut self,
        expression: &Atom,
        active: &Atom,
        dimension: i32,
        denominator: bool,
        label: String,
        indices: &mut HashMap<Atom, usize>,
    ) {
        if expression.is_zero() && !denominator {
            return;
        }
        // Symbolica performs the algebra. This upper bound is only structural
        // resource accounting; it does not construct polynomial coefficients.
        let expanded = if term_budget(expression.as_view()) <= MAX_LOCAL_TERMS {
            expression.expand()
        } else {
            expression.clone()
        };
        let terms: Vec<_> = match expanded.as_view() {
            AtomView::Add(sum) => sum.iter().map(|term| term.to_owned()).collect(),
            _ => vec![expanded.clone()],
        };
        if terms.len() < 2 && !denominator {
            return;
        }
        let operations = operation_count(expression.as_view()).max(1);
        let value = self.intern(guarded(active, expression.clone(), denominator), indices);
        let terms = terms
            .into_iter()
            .map(|term| self.intern(guarded(active, term, false), indices))
            .collect();
        self.checks.push(Check {
            value,
            terms,
            dimension,
            operations,
            denominator,
            label,
        });
    }

    /// Inspect one packed sample. Dual components remain a separate axis.
    ///
    /// Denominator branch selection uses the primal. Additive cancellation is
    /// checked independently in every Taylor coefficient, with a mixed
    /// absolute/relative criterion in the coefficient's physical mass units.
    #[cfg(test)]
    pub(crate) fn inspect_lane<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
    ) -> Option<String> {
        self.inspect_lane_with_accuracy(
            values,
            input,
            components,
            prototype,
            prototype.get_precision(),
        )
    }

    /// Check a requested accuracy independently of the arithmetic precision.
    /// This lets a promoted workspace retain the original binary64 target.
    pub(crate) fn inspect_lane_with_accuracy<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
    ) -> Option<String> {
        self.inspect_lane_with_accuracy_impl::<T, false>(
            values,
            input,
            components,
            prototype,
            accuracy_bits,
            None,
        )
    }

    pub(crate) fn inspect_lane_with_accuracy_context<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        context: &mut GuardContext<T>,
    ) -> Option<String> {
        self.inspect_lane_with_accuracy_impl::<T, true>(
            values,
            input,
            components,
            prototype,
            accuracy_bits,
            Some(context),
        )
    }

    fn inspect_lane_with_accuracy_impl<T: NativeFloat, const CACHED: bool>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        mut context: Option<&mut GuardContext<T>>,
    ) -> Option<String> {
        if components == 0 || values.len() != self.expressions.len() * components {
            return Some("invalid stability-monitor buffer".into());
        }
        let precision = prototype.get_precision();
        let mut roundoff = if CACHED {
            OperandRoundoff::with_unit(
                prototype,
                context.as_deref().unwrap().prototype_unit.clone(),
            )
        } else {
            OperandRoundoff::new(prototype)
        };
        // At binary64 this is 2^-34, below the 1e-10 validation target.
        // Additional precision tightens the criterion without an f64 detour.
        let tolerance = if CACHED {
            context.as_deref().unwrap().tolerance.clone()
        } else {
            dyadic(prototype, 19i64 - i64::from(accuracy_bits))
        };
        let scale = if CACHED {
            None
        } else {
            Some(self.kinematic_scale(input, components, prototype))
        };
        for (check_index, check) in self.checks.iter().enumerate() {
            let count = if check.denominator { 1 } else { components };
            let operation_factor = prototype.from_usize(check.operations.saturating_mul(8));
            for component in 0..count {
                let value = &values[check.value * components + component];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Some(format!("nonfinite stability monitor for {}", check.label));
                }
                let magnitude = NativeFloat::hypot(&value.re, &value.im);
                if check.denominator && magnitude.is_zero() {
                    return Some(format!("zero recurrence denominator: {}", check.label));
                }
                let mut contribution_error = prototype.zero();
                for &index in &check.terms {
                    let term = &values[index * components + component];
                    if !term.re.is_finite() || !term.im.is_finite() {
                        return Some(format!(
                            "nonfinite stability contribution for {}",
                            check.label
                        ));
                    }
                    let estimate = match roundoff.complex(term) {
                        Ok(value) => value,
                        Err(reason) => return Some(format!("{reason} for {}", check.label)),
                    };
                    contribution_error += estimate;
                }
                let reference = if check.denominator {
                    magnitude
                } else {
                    let natural = if CACHED {
                        let context = context.as_deref_mut().unwrap();
                        context.natural(
                            context.local_slots[check_index],
                            self,
                            input,
                            components,
                            prototype,
                        )
                    } else {
                        signed_power(scale.as_ref().unwrap(), check.dimension)
                    };
                    if !natural.is_finite() || natural.is_zero() {
                        return Some(format!(
                            "conditioning normalization is outside the numeric range for {}",
                            check.label
                        ));
                    }
                    if magnitude > natural {
                        magnitude
                    } else {
                        natural
                    }
                };
                let estimate = operation_factor.clone() * contribution_error;
                let target = tolerance.clone() * &reference;
                if !target.is_finite() || (target.is_zero() && !reference.is_zero()) {
                    return Some(format!(
                        "conditioning tolerance is outside the numeric range for {}",
                        check.label
                    ));
                }
                if !estimate.is_finite() || estimate > target {
                    return Some(format!(
                        "estimated cancellation exceeds 2^(19-{accuracy_bits}) with operand-dependent roundoff at configured precision {precision} in {}, Taylor component {component}; this estimate is not a certified error bound",
                        check.label,
                    ));
                }
            }
        }
        None
    }

    /// Estimate the sensitivity to one working-precision unit of perturbation
    /// in every scalar master. Shared recurrence paths propagate absolute
    /// amplitudes, so earlier cancellations remain visible at later nodes.
    /// This is a primal sensitivity estimate, not a bound on backend errors or
    /// on compounded Taylor-component errors; local Dual checks remain active.
    pub(crate) fn inspect_primal_lane<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        prototype: &T,
        accuracy_bits: u32,
    ) -> Option<String> {
        self.inspect_primal_lane_impl::<T, false>(values, input, prototype, accuracy_bits, None)
    }

    pub(crate) fn inspect_primal_lane_context<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        prototype: &T,
        accuracy_bits: u32,
        context: &mut GuardContext<T>,
    ) -> Option<String> {
        self.inspect_primal_lane_impl::<T, true>(
            values,
            input,
            prototype,
            accuracy_bits,
            Some(context),
        )
    }

    fn inspect_primal_lane_impl<T: NativeFloat, const CACHED: bool>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        prototype: &T,
        accuracy_bits: u32,
        mut context: Option<&mut GuardContext<T>>,
    ) -> Option<String> {
        if values.len() != self.primal_expressions.len() {
            return Some("invalid primal stability-monitor buffer".into());
        }
        let precision = prototype.get_precision();
        let mut roundoff = if CACHED {
            OperandRoundoff::with_unit(
                prototype,
                context.as_deref().unwrap().prototype_unit.clone(),
            )
        } else {
            OperandRoundoff::new(prototype)
        };
        let tolerance = if CACHED {
            context.as_deref().unwrap().tolerance.clone()
        } else {
            dyadic(prototype, 19i64 - i64::from(accuracy_bits))
        };
        let scale = if CACHED {
            None
        } else {
            Some(self.kinematic_scale(input, 1, prototype))
        };
        for (check_index, check) in self.primal_checks.iter().enumerate() {
            let value = &values[check.value];
            let amplitude = &values[check.amplitude];
            if !value.re.is_finite()
                || !value.im.is_finite()
                || !amplitude.re.is_finite()
                || !amplitude.im.is_finite()
            {
                return Some(format!(
                    "nonfinite primal master-amplitude monitor for {}",
                    check.label
                ));
            }
            let magnitude = NativeFloat::hypot(&value.re, &value.im);
            let natural = if CACHED {
                let context = context.as_deref_mut().unwrap();
                context.natural(context.primal_slots[check_index], self, input, 1, prototype)
            } else {
                signed_power(scale.as_ref().unwrap(), check.dimension)
            };
            if !natural.is_finite() || natural.is_zero() {
                return Some(format!(
                    "primal conditioning normalization is outside the numeric range for {}",
                    check.label
                ));
            }
            let reference = if magnitude > natural {
                magnitude
            } else {
                natural
            };
            let target = tolerance.clone() * reference;
            let estimate = match roundoff.complex(amplitude) {
                Ok(value) => value,
                Err(reason) => return Some(format!("{reason} for {}", check.label)),
            };
            if !target.is_finite() || target.is_zero() {
                return Some(format!(
                    "primal conditioning tolerance is outside the numeric range for {}",
                    check.label
                ));
            }
            if !estimate.is_finite() || estimate > target {
                return Some(format!(
                    "estimated compounded master cancellation exceeds 2^(19-{accuracy_bits}) with operand-dependent roundoff at configured precision {precision} in {}; primal-only sensitivity estimate, not a certified error bound",
                    check.label
                ));
            }
        }
        None
    }

    /// Check the precision retained by every returned coefficient, including
    /// bare scalar masters and monomials without an additive recurrence check.
    /// Tracked cancellation loss is scaled by the signed output's magnitude,
    /// never by the upstream contribution envelope a second time.
    pub(crate) fn inspect_output_precision_lane<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
    ) -> Option<String> {
        self.inspect_output_precision_lane_impl::<T, false>(
            values,
            input,
            components,
            prototype,
            accuracy_bits,
            None,
        )
    }

    pub(crate) fn inspect_output_precision_lane_context<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        context: &mut GuardContext<T>,
    ) -> Option<String> {
        self.inspect_output_precision_lane_impl::<T, true>(
            values,
            input,
            components,
            prototype,
            accuracy_bits,
            Some(context),
        )
    }

    fn inspect_output_precision_lane_impl<T: NativeFloat, const CACHED: bool>(
        &self,
        values: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        mut context: Option<&mut GuardContext<T>>,
    ) -> Option<String> {
        if components == 0 || values.len() != self.output_dimensions.len() * components {
            return Some("invalid output-precision buffer".into());
        }
        if T::FIXED_BITS == Some(53)
            && accuracy_bits <= prototype.get_precision()
            && values
                .iter()
                .flat_map(|value| [&value.re, &value.im])
                .all(|part| {
                    matches!(
                        part.real_classify(),
                        Some(FpCategory::Normal | FpCategory::Zero)
                    )
                })
        {
            // With normal binary64 components, one output ULP is at
            // least 18 bits below the configured dimensional target. Existing
            // contribution checks still inspect cancellation; no repeated
            // norm, power or precision-map work is needed here.
            return None;
        }
        let mut roundoff = if CACHED {
            OperandRoundoff::with_unit(
                prototype,
                context.as_deref().unwrap().prototype_unit.clone(),
            )
        } else {
            OperandRoundoff::new(prototype)
        };
        let tolerance = if CACHED {
            context.as_deref().unwrap().tolerance.clone()
        } else {
            dyadic(prototype, 19i64 - i64::from(accuracy_bits))
        };
        let scale = if CACHED {
            None
        } else {
            Some(self.kinematic_scale(input, components, prototype))
        };
        for (output, &dimension) in self.output_dimensions.iter().enumerate() {
            for component in 0..components {
                let value = &values[output * components + component];
                if value.re.is_zero() && value.im.is_zero() {
                    // Active-master argument provenance is checked separately:
                    // a rounded scalar zero must not become an exact-zero proof.
                    continue;
                }
                let estimate = match roundoff.complex(value) {
                    Ok(value) => value,
                    Err(reason) => {
                        return Some(format!(
                            "{reason} in output {output}, Taylor component {component}"
                        ));
                    }
                };
                let natural = if CACHED {
                    let context = context.as_deref_mut().unwrap();
                    context.natural(
                        context.output_slots[output],
                        self,
                        input,
                        components,
                        prototype,
                    )
                } else {
                    signed_power(scale.as_ref().unwrap(), dimension)
                };
                if !natural.is_finite() || natural.is_zero() {
                    return Some(format!(
                        "output-precision normalization is outside the numeric range in output {output}"
                    ));
                }
                let magnitude = NativeFloat::hypot(&value.re, &value.im);
                let reference = if magnitude > natural {
                    magnitude
                } else {
                    natural
                };
                let target = tolerance.clone() * reference;
                if !target.is_finite() || target.is_zero() {
                    return Some(format!(
                        "output-precision tolerance is outside the numeric range in output {output}"
                    ));
                }
                if estimate > target {
                    return Some(format!(
                        "tracked coefficient precision cannot support 2^(19-{accuracy_bits}) in output {output}, Taylor component {component}; this estimate is not a certified error bound"
                    ));
                }
            }
        }
        None
    }

    fn kinematic_scale<T: NativeFloat>(
        &self,
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
    ) -> T {
        // Include routing coordinates because the API
        // returns Cartesian moments in that routing, not only invariants.
        // Use mu only for a genuinely scaleless topology: changing an
        // otherwise arbitrary renormalization scale must not loosen the
        // absolute conditioning test for a massive tensor coefficient.
        let mut scale = prototype.zero();
        for index in self.layout.shifts.clone() {
            let value = &input[index * components];
            let magnitude = NativeFloat::hypot(&value.re, &value.im);
            if magnitude > scale {
                scale = magnitude;
            }
        }
        for index in self
            .layout
            .masses
            .clone()
            .chain(self.layout.scalar_products.clone().into_iter().flatten())
        {
            let value = &input[index * components];
            let magnitude = NativeFloat::hypot(&value.re, &value.im).sqrt();
            if magnitude > scale {
                scale = magnitude;
            }
        }
        if scale.is_zero() {
            input[self.layout.scale * components].re.clone()
        } else {
            scale
        }
    }
}

fn depends_on_amplitudes(expression: &Atom, amplitudes: &HashMap<Atom, Atom>) -> bool {
    expression
        .get_all_symbols(false)
        .into_iter()
        .any(|symbol| amplitudes.contains_key(&Atom::var(symbol)))
}

/// Symbolica extracts the affine coefficients in immediate recurrence inputs.
/// Aliases are never substituted recursively. Conditional subexpressions are
/// polynomial generators of their own, with separately constructed lazy
/// amplitudes, preserving branches even when IF occurs below a sum/product.
fn master_amplitude(
    expression: &Atom,
    amplitudes: &HashMap<Atom, Atom>,
) -> Result<Atom, ReductionError> {
    if let Some(amplitude) = amplitudes.get(expression) {
        return Ok(amplitude.clone());
    }
    if !depends_on_amplitudes(expression, amplitudes) {
        return Ok(Symbol::ABS.call((expression,)));
    }
    if let Some((condition, yes, no)) = if_parts(expression.as_view()) {
        return Ok(Symbol::IF.call((
            condition,
            master_amplitude(&yes.to_owned(), amplitudes)?,
            master_amplitude(&no.to_owned(), amplitudes)?,
        )));
    }
    let mut variables = BTreeSet::new();
    collect_amplitude_variables(expression.as_view(), amplitudes, &mut variables);
    let variables: Vec<_> = variables.into_iter().collect();
    let variable_amplitudes = variables
        .iter()
        .map(|variable| master_amplitude(variable, amplitudes))
        .collect::<Result<Vec<_>, _>>()?;
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    };
    let polynomial = expression.to_polynomial_in_vars_with_field::<u16>(&variables, &field);
    let mut result = Atom::new();
    for term in &polynomial {
        if depends_on_amplitudes(term.coefficient, amplitudes)
            || term
                .exponents
                .iter()
                .map(|&power| u32::from(power))
                .sum::<u32>()
                > 1
        {
            return Err(ReductionError::Preparation(format!(
                "primal stability monitor requires affine scalar-master dependence: {expression}"
            )));
        }
        let coefficient = Symbol::ABS.call((term.coefficient,));
        if let Some(index) = term.exponents.iter().position(|&power| power == 1) {
            result = result + coefficient * &variable_amplitudes[index];
        } else {
            result += coefficient;
        }
    }
    Ok(result)
}

fn collect_amplitude_variables(
    expression: AtomView<'_>,
    amplitudes: &HashMap<Atom, Atom>,
    variables: &mut BTreeSet<Atom>,
) {
    if if_parts(expression).is_some() {
        let value = expression.to_owned();
        if depends_on_amplitudes(&value, amplitudes) {
            variables.insert(value);
        }
        return;
    }
    match expression {
        AtomView::Var(_) => {
            let value = expression.to_owned();
            if amplitudes.contains_key(&value) {
                variables.insert(value);
            }
        }
        AtomView::Add(sum) => {
            for term in sum {
                collect_amplitude_variables(term, amplitudes, variables);
            }
        }
        AtomView::Mul(product) => {
            for term in product {
                collect_amplitude_variables(term, amplitudes, variables);
            }
        }
        AtomView::Pow(power) => {
            collect_amplitude_variables(power.get_base(), amplitudes, variables);
            collect_amplitude_variables(power.get_exp(), amplitudes, variables);
        }
        AtomView::Fun(function) => {
            for term in function {
                collect_amplitude_variables(term, amplitudes, variables);
            }
        }
        AtomView::Num(_) => {}
    }
}

/// Weighted roundoff estimates from the precision of actual operands.
///
/// Exact-zero component metadata is irrelevant. A tiny weak imaginary part
/// must not set the relative uncertainty of a large strong real part. Float's
/// loose precision tracking can overestimate by two bits; reserve three bits
/// for dynamically tracked domains. Fixed-precision domains keep their
/// previous unit-roundoff convention.
struct OperandRoundoff<'a, T: NativeFloat> {
    prototype: &'a T,
    prototype_bits: u32,
    prototype_unit: T,
    units: HashMap<u32, T>,
}

impl<'a, T: NativeFloat> OperandRoundoff<'a, T> {
    fn with_unit(prototype: &'a T, prototype_unit: T) -> Self {
        Self {
            prototype,
            prototype_bits: prototype.get_precision(),
            prototype_unit,
            units: HashMap::new(),
        }
    }

    fn new(prototype: &'a T) -> Self {
        let prototype_bits = prototype.get_precision();
        let margin = if prototype.fixed_precision() { 0 } else { 3 };
        Self {
            prototype,
            prototype_bits,
            prototype_unit: dyadic(prototype, 1 + margin - i64::from(prototype_bits)),
            units: HashMap::new(),
        }
    }

    fn component(&mut self, value: &T) -> Result<T, &'static str> {
        if value.is_zero() {
            return Ok(self.prototype.zero());
        }
        if !value.is_finite() {
            return Err("nonfinite operand in roundoff estimate");
        }
        let bits = self.prototype_bits.min(value.get_precision());
        if bits == 0 {
            return Err("operand roundoff estimate requires positive precision");
        }
        let margin = if self.prototype.fixed_precision() {
            0
        } else {
            3
        };
        let unit = if bits == self.prototype_bits {
            &self.prototype_unit
        } else {
            self.units
                .entry(bits)
                .or_insert_with(|| dyadic(self.prototype, 1 + margin - i64::from(bits)))
        };
        if !unit.is_finite() {
            return Err("operand roundoff unit is outside the supported numeric range");
        }
        let magnitude = value.norm();
        let estimate = unit.clone() * &magnitude;
        if !estimate.is_finite() {
            return Err("nonfinite operand roundoff estimate");
        }
        if unit.is_zero()
            || estimate.is_zero()
            || magnitude.real_classify() == Some(FpCategory::Subnormal)
            || estimate.real_classify() == Some(FpCategory::Subnormal)
        {
            if *unit > self.prototype.one() {
                return Err(
                    "operand roundoff underflow fallback requires a unit no larger than one",
                );
            }
            // The relative model does not cover gradual underflow. For
            // epsilon <= 1, |x| is a deliberately loose upper estimate of
            // epsilon*|x|. Keeping it positive prevents a false-safe zero;
            // comparison with the dimensional target can still accept a
            // harmless tiny contribution. This is not an interval bound.
            return Ok(magnitude);
        }
        Ok(estimate)
    }

    fn complex(&mut self, value: &Complex<T>) -> Result<T, &'static str> {
        if value.re.is_zero() && value.im.is_zero() {
            return Ok(self.prototype.zero());
        }
        if [&value.re, &value.im].into_iter().all(|part| {
            part.is_zero()
                || (part.is_finite()
                    && part.get_precision() >= self.prototype_bits
                    && part.real_classify() != Some(FpCategory::Subnormal))
        }) {
            // Uniform f64/DD/Float operands reuse the cached unit and the
            // previous Euclidean norm; no per-term rational construction or
            // cache allocation is needed on this common path.
            let magnitude = NativeFloat::hypot(&value.re, &value.im);
            let estimate = self.prototype_unit.clone() * magnitude;
            if estimate.is_finite()
                && !estimate.is_zero()
                && estimate.real_classify() != Some(FpCategory::Subnormal)
                && self.prototype_unit <= self.prototype.one()
            {
                return Ok(estimate);
            }
        }
        let re = self.component(&value.re)?;
        let im = self.component(&value.im)?;
        let estimate = NativeFloat::hypot(&re, &im);
        if !estimate.is_finite() || (estimate.is_zero() && (!re.is_zero() || !im.is_zero())) {
            Err("complex operand roundoff estimate is outside the numeric range")
        } else {
            Ok(estimate)
        }
    }
}

/// Scalar internals can round a nonzero finite part to zero, erasing its
/// precision provenance before output/amplitude checks. Inspect only actual
/// active requests, including their Taylor coefficients, to retain that
/// provenance without penalizing unrelated routing inputs. This conservative
/// gate may request rescue for exact coincidences, tiny weak seeds, or an
/// analytically unused scale; it never lowers the requested accuracy target.
pub(crate) fn master_argument_precision_failure<T: NativeFloat>(
    arguments: &[Complex<T>],
    original_scale: &[Complex<T>],
    components: usize,
    prototype: &T,
    accuracy_bits: u32,
) -> Option<String> {
    if prototype.fixed_precision() {
        return None;
    }
    if components == 0 || arguments.len() < components {
        return Some("invalid scalar-master precision provenance buffer".into());
    }
    // Require the entire non-scale jet to vanish. A scaleless primal with a
    // transverse seed is not the constant scaleless analytic branch.
    if arguments[..arguments.len() - components]
        .iter()
        .all(|value| value.re.is_zero() && value.im.is_zero())
    {
        return None;
    }
    // The public API supplies mu, whereas masters consume mu². A weak Taylor
    // coefficient can vanish in that nonlinear transformation; preserve its
    // original precision even when the transformed coefficient is zero.
    let bits = arguments
        .iter()
        .chain(original_scale)
        .flat_map(|value| [&value.re, &value.im])
        .filter(|value| !value.is_zero())
        .fold(prototype.get_precision(), |bits, value| {
            bits.min(value.get_precision())
        });
    // 8*2^(1-bits) <= 2^(19-accuracy_bits), retaining the existing
    // 19-bit requested-error margin and three tracked-precision guard bits.
    if bits == 0 || u64::from(bits) + 15 < u64::from(accuracy_bits) {
        Some(format!(
            "active scalar-master arguments retain {bits} bits, insufficient for the requested 2^(19-{accuracy_bits}) error budget; conservative precision provenance check requires rescue"
        ))
    } else {
        None
    }
}

fn dyadic<T: NativeFloat>(prototype: &T, exponent: i64) -> T {
    let power = Integer::from(2).pow(exponent.unsigned_abs());
    let value = if exponent < 0 {
        Rational::from((Integer::one(), power))
    } else {
        Rational::from(power)
    };
    prototype.from_rational(&value)
}

fn signed_power<T: NativeFloat>(value: &T, exponent: i32) -> T {
    let power = value.pow(u64::from(exponent.unsigned_abs()));
    if exponent < 0 { power.inv() } else { power }
}

fn integer_exponent(value: AtomView<'_>) -> Option<i64> {
    if let AtomView::Num(number) = value {
        if let CoefficientView::Natural(n, 1, 0, _) = number.get_coeff_view() {
            return Some(n);
        }
    }
    None
}

fn if_parts(value: AtomView<'_>) -> Option<(AtomView<'_>, AtomView<'_>, AtomView<'_>)> {
    if let AtomView::Fun(function) = value {
        if function.get_symbol() == Symbol::IF {
            let mut args = function.iter();
            let parts = (args.next()?, args.next()?, args.next()?);
            if args.next().is_none() {
                return Some(parts);
            }
        }
    }
    None
}

fn guarded(active: &Atom, value: Atom, inactive_one: bool) -> Atom {
    if active == &Atom::num(1) {
        return value;
    }
    Symbol::IF.call((active, value, Atom::num(i32::from(inactive_one))))
}

fn branch_path(active: &Atom, condition: AtomView<'_>, yes: bool) -> Atom {
    let selected = Symbol::IF.call((
        condition,
        Atom::num(i32::from(yes)),
        Atom::num(i32::from(!yes)),
    ));
    guarded(active, selected, false)
}

fn union_paths(paths: BTreeSet<Atom>) -> Atom {
    if paths.contains(&Atom::num(1)) {
        return Atom::num(1);
    }
    paths.into_iter().fold(Atom::new(), |union, path| {
        Symbol::IF.call((path, Atom::num(1), union))
    })
}

fn parameter_only(
    expression: &Atom,
    definitions: &HashMap<Atom, Atom>,
    masters: &BTreeSet<Atom>,
    safe_aliases: &BTreeSet<Atom>,
) -> bool {
    expression.get_all_symbols(false).into_iter().all(|symbol| {
        let variable = Atom::var(symbol);
        !masters.contains(&variable)
            && (!definitions.contains_key(&variable) || safe_aliases.contains(&variable))
    })
}

fn intern_activation_predicate(
    expression: Atom,
    aliases: &mut Vec<(Atom, Atom)>,
    indices: &mut HashMap<Atom, Atom>,
) -> Atom {
    intern_named_predicate(expression, "iregi::master_active_path", aliases, indices)
}

fn intern_named_predicate(
    expression: Atom,
    namespace: &str,
    aliases: &mut Vec<(Atom, Atom)>,
    indices: &mut HashMap<Atom, Atom>,
) -> Atom {
    if expression == Atom::num(1) || expression.is_zero() {
        return expression;
    }
    if let Some(alias) = indices.get(&expression) {
        return alias.clone();
    }
    let alias = Atom::var(symbolica::symbol!(format!("{namespace}_{}", aliases.len())));
    indices.insert(expression.clone(), alias.clone());
    aliases.push((alias.clone(), expression));
    alias
}

fn visit_if_branches<'a>(
    value: AtomView<'a>,
    active: &Atom,
    visit: &mut impl FnMut(AtomView<'a>, &Atom),
) {
    if let Some((condition, yes, no)) = if_parts(value) {
        visit(yes, &branch_path(active, condition, true));
        visit(no, &branch_path(active, condition, false));
        return;
    }
    match value {
        AtomView::Pow(power) => visit_if_branches(power.get_base(), active, visit),
        AtomView::Add(sum) => {
            for term in sum {
                visit_if_branches(term, active, visit);
            }
        }
        AtomView::Mul(product) => {
            for term in product {
                visit_if_branches(term, active, visit);
            }
        }
        AtomView::Fun(function) => {
            for term in function {
                visit_if_branches(term, active, visit);
            }
        }
        _ => {}
    }
}

fn collect_alias_uses(
    value: AtomView<'_>,
    active: &Atom,
    definitions: &HashMap<Atom, Atom>,
    uses: &mut HashMap<Atom, BTreeSet<Atom>>,
) {
    if let Some((condition, yes, no)) = if_parts(value) {
        collect_alias_uses(condition, active, definitions, uses);
        collect_alias_uses(
            yes,
            &branch_path(active, condition, true),
            definitions,
            uses,
        );
        collect_alias_uses(
            no,
            &branch_path(active, condition, false),
            definitions,
            uses,
        );
        return;
    }
    match value {
        AtomView::Var(_) => {
            let alias = value.to_owned();
            if definitions.contains_key(&alias) {
                uses.entry(alias).or_default().insert(active.clone());
            }
        }
        AtomView::Pow(power) => {
            for term in power {
                collect_alias_uses(term, active, definitions, uses);
            }
        }
        AtomView::Add(sum) => {
            for term in sum {
                collect_alias_uses(term, active, definitions, uses);
            }
        }
        AtomView::Mul(product) => {
            for term in product {
                collect_alias_uses(term, active, definitions, uses);
            }
        }
        AtomView::Fun(function) => {
            for term in function {
                collect_alias_uses(term, active, definitions, uses);
            }
        }
        _ => {}
    }
}

fn collect_denominators(
    value: AtomView<'_>,
    active: &Atom,
    denominators: &mut BTreeSet<(Atom, Atom)>,
) {
    if let Some((condition, yes, no)) = if_parts(value) {
        collect_denominators(condition, active, denominators);
        collect_denominators(yes, &branch_path(active, condition, true), denominators);
        collect_denominators(no, &branch_path(active, condition, false), denominators);
        return;
    }
    match value {
        AtomView::Pow(power) => {
            if integer_exponent(power.get_exp()).is_some_and(|n| n < 0) {
                denominators.insert((power.get_base().to_owned(), active.clone()));
            }
            collect_denominators(power.get_base(), active, denominators);
        }
        AtomView::Add(sum) => {
            for term in sum {
                collect_denominators(term, active, denominators);
            }
        }
        AtomView::Mul(product) => {
            for term in product {
                collect_denominators(term, active, denominators);
            }
        }
        AtomView::Fun(function) => {
            for term in function {
                collect_denominators(term, active, denominators);
            }
        }
        _ => {}
    }
}

// Physical-dimension metadata and resource counts do not manipulate algebra.
fn mass_dimension(value: AtomView<'_>, dimensions: &HashMap<Atom, i32>) -> Option<i32> {
    if let Some((_, yes, no)) = if_parts(value) {
        if yes.is_zero() {
            return mass_dimension(no, dimensions);
        }
        if no.is_zero() {
            return mass_dimension(yes, dimensions);
        }
        let left = mass_dimension(yes, dimensions)?;
        let right = mass_dimension(no, dimensions)?;
        return (left == right).then_some(left);
    }
    match value {
        AtomView::Num(_) | AtomView::Fun(_) => Some(0),
        AtomView::Var(_) => dimensions.get(&value.to_owned()).copied().or(Some(0)),
        AtomView::Pow(power) => mass_dimension(power.get_base(), dimensions)?
            .checked_mul(integer_exponent(power.get_exp())?.try_into().ok()?),
        AtomView::Mul(product) => product.iter().try_fold(0i32, |total, term| {
            total.checked_add(mass_dimension(term, dimensions)?)
        }),
        AtomView::Add(sum) => {
            let mut result = None;
            for term in sum {
                if term.is_zero() {
                    continue;
                }
                let dimension = mass_dimension(term, dimensions)?;
                if result.is_some_and(|other| other != dimension) {
                    return None;
                }
                result = Some(dimension);
            }
            result.or(Some(0))
        }
    }
}

fn operation_count(value: AtomView<'_>) -> usize {
    match value {
        AtomView::Add(sum) => sum.iter().map(operation_count).sum::<usize>() + sum.iter().count(),
        AtomView::Mul(product) => {
            product.iter().map(operation_count).sum::<usize>() + product.iter().count()
        }
        AtomView::Pow(power) => operation_count(power.get_base()) + 2,
        AtomView::Fun(function) => function.iter().map(operation_count).sum::<usize>() + 8,
        _ => 0,
    }
}

fn term_budget(value: AtomView<'_>) -> usize {
    let cap = MAX_LOCAL_TERMS + 1;
    match value {
        AtomView::Add(sum) => sum.iter().fold(0usize, |n, term| {
            n.saturating_add(term_budget(term)).min(cap)
        }),
        AtomView::Mul(product) => product.iter().fold(1usize, |n, term| {
            n.saturating_mul(term_budget(term)).min(cap)
        }),
        AtomView::Pow(power) => match integer_exponent(power.get_exp()) {
            Some(n) if n > 0 => term_budget(power.get_base())
                .saturating_pow(n.min(cap as i64) as u32)
                .min(cap),
            _ => 1,
        },
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::float::{Float, FloatLike, Real, SingleFloat};

    fn synthetic(dimension: i32, denominator: bool) -> StabilityPlan {
        StabilityPlan {
            expressions: vec![Atom::new(); 3],
            predicate_aliases: Vec::new(),
            master_activations: Vec::new(),
            master_activation_aliases: Vec::new(),
            primal_expressions: Vec::new(),
            primal_aliases: Vec::new(),
            primal_checks: Vec::new(),
            snapping_expressions: Vec::new(),
            snapping_aliases: Vec::new(),
            snapping_labels: Vec::new(),
            snapping_threshold: None,
            snapping_zero_image: None,
            checks: vec![Check {
                value: 0,
                terms: vec![1, 2],
                dimension,
                operations: 3,
                denominator,
                label: "synthetic cancellation".into(),
            }],
            output_dimensions: Vec::new(),
            layout: InputLayout {
                shifts: 0..0,
                scalar_products: None,
                masses: 0..0,
                scale: 0,
            },
        }
    }

    fn complex<T: NativeFloat>(value: T) -> Complex<T> {
        Complex::new(value.clone(), value.zero())
    }

    fn evaluate_snapping(
        monitor: &StabilityPlan,
        parameters: &[Atom],
        rows: &[[f64; 6]],
    ) -> Vec<Vec<f64>> {
        use symbolica::evaluate::ExpressionEvaluator;
        let mut expressions = monitor.snapping_expressions().to_vec();
        expressions.extend(monitor.snapping_threshold().cloned());
        expressions.extend(monitor.snapping_zero_image().cloned());
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&expressions, parameters)
                .add_aliases(monitor.snapping_aliases().iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        let mut evaluator = exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        0.0f64.from_rational(&value.re),
                        0.0f64.from_rational(&value.im),
                    )
                },
                53,
            )
            .into_batch_evaluator();
        let input: Vec<_> = rows.iter().flatten().copied().map(complex).collect();
        let mut output = vec![complex(0.0); rows.len() * expressions.len()];
        evaluator
            .evaluate_batch(&input, &mut output, rows.len())
            .unwrap();
        output
            .chunks_exact(expressions.len())
            .map(|row| row.iter().map(|value| value.re).collect())
            .collect()
    }

    #[test]
    fn snapping_boundaries_follow_query_demand_and_skip_discarded_queries() {
        use crate::{Algorithm, ReductionSettings, snapping::LegacySnapping};
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let mut snapping = LegacySnapping::new(&plan.parameters[4..5], "1/8").unwrap();
        let used = snapping.snap_scalar_product(plan.parameters[0].clone(), "used query");
        let _unused =
            snapping.snap_scalar_product(Atom::num(Rational::from((1, 2))), "discarded query");
        plan.aliases = snapping.aliases().to_vec();
        plan.snapping_aliases = plan.aliases.clone();
        plan.snapping_diagnostics = snapping.diagnostics().to_vec();
        plan.snapping_threshold = Some(Atom::num(Rational::from((1, 8))));
        plan.moments = vec![[
            Symbol::IF.call((&plan.parameters[1], used, Atom::new())),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ]];
        let monitor = StabilityPlan::prepare(&plan).unwrap();
        assert!(
            monitor
                .snapping_labels()
                .iter()
                .any(|label| label.contains("used query"))
        );
        assert!(
            !monitor
                .snapping_labels()
                .iter()
                .any(|label| label.contains("discarded query"))
        );
        let values = evaluate_snapping(
            &monitor,
            &plan.parameters,
            &[
                [0.5, 0.0, 0.0, 0.0, 4.0, 3.0],
                [0.5, 1.0, 0.0, 0.0, 4.0, 3.0],
                [2.0, 1.0, 0.0, 0.0, 4.0, 3.0],
            ],
        );
        assert!(values[0].iter().all(|&value| value == 1.0));
        assert!(values[1].contains(&0.0));
        assert!(
            values[2]
                .iter()
                .all(|value| value.is_finite() && *value != 0.0)
        );
        assert_eq!(*values[1].last().unwrap(), 0.125);
    }

    #[test]
    fn snapping_used_only_by_active_master_arguments_is_monitored() {
        use crate::{Algorithm, ReductionSettings, snapping::LegacySnapping};
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let mut snapping = LegacySnapping::new(&plan.parameters[4..5], "1/8").unwrap();
        let query = snapping.snap_scalar_product(plan.parameters[0].clone(), "master argument");
        plan.aliases = snapping.aliases().to_vec();
        plan.snapping_aliases = plan.aliases.clone();
        plan.snapping_diagnostics = snapping.diagnostics().to_vec();
        plan.snapping_threshold = Some(Atom::num(Rational::from((1, 8))));
        plan.master_arguments[0][0] = query;
        plan.moments = vec![[
            Symbol::IF.call((&plan.parameters[1], &plan.leaves[0].slots[0], Atom::new())),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ]];
        let monitor = StabilityPlan::prepare(&plan).unwrap();
        assert!(
            monitor
                .snapping_labels()
                .iter()
                .any(|label| label.contains("master argument"))
        );
        let values = evaluate_snapping(
            &monitor,
            &plan.parameters,
            &[
                [0.5, 0.0, 0.0, 0.0, 4.0, 3.0],
                [0.5, 1.0, 0.0, 0.0, 4.0, 3.0],
            ],
        );
        assert!(values[0].iter().all(|&value| value == 1.0));
        assert!(values[1].contains(&0.0));
    }

    #[test]
    fn snapping_diagonal_probe_preserves_unused_scalar_tadpoles() {
        use crate::{Algorithm, InputPolicy, ReductionSettings};
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::Ibp;
        settings.input_policy = InputPolicy::LegacyOnShell {
            threshold: "0".into(),
        };
        let scalar = ReductionPlan::new(settings.clone()).unwrap();
        let unused = StabilityPlan::prepare(&scalar).unwrap();
        assert!(unused.snapping_expressions().is_empty());
        assert!(unused.snapping_threshold().is_none());
        assert!(unused.snapping_zero_image().is_none());

        settings.rank = 2;
        let tensor = ReductionPlan::new(settings).unwrap();
        assert!(tensor.snapping_zero_image_required);
        let required = StabilityPlan::prepare(&tensor).unwrap();
        assert!(required.snapping_threshold().is_some());
        assert!(required.snapping_zero_image().is_some());
        let values = evaluate_snapping(
            &required,
            &tensor.parameters,
            &[
                [0.0, 0.0, 0.0, 0.0, -4.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
            ],
        );
        assert_eq!(*values[0].last().unwrap(), -4.0);
        assert_eq!(*values[1].last().unwrap(), 0.0);
    }

    #[test]
    fn master_amplitudes_keep_nested_branches_and_reject_nonlinear_masters() {
        let x = Atom::var(symbolica::symbol!("iregi_test::amplitude_master"));
        let p = Atom::var(symbolica::symbol!("iregi_test::amplitude_parameter"));
        let amplitudes = HashMap::from([(x.clone(), Symbol::ABS.call((&x,)))]);
        let conditional = Symbol::IF.call((&p, &x, Atom::num(2) * &x));
        let expression = Atom::num(3) * &conditional + Atom::num(7);
        let expected = Atom::num(3)
            * Symbol::IF.call((
                &p,
                Symbol::ABS.call((&x,)),
                Atom::num(2) * Symbol::ABS.call((&x,)),
            ))
            + Atom::num(7);
        assert_eq!(
            master_amplitude(&expression, &amplitudes).unwrap(),
            expected
        );
        assert!(master_amplitude(&x.pow(2), &amplitudes).is_err());
        assert!(master_amplitude(&Symbol::EXP.call((&x,)), &amplitudes).is_err());
    }

    #[test]
    fn pentagon_master_cancellation_triggers_precision_rescue() {
        use crate::{
            Algorithm, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace, ReductionSettings,
            RescueSettings, basis::Monomial, convention::convert_convention, pv::PvCompiler,
        };

        // The independently preserved 128/256-bit probe in
        // validation/pv-high-rank/precision-probe.json fixes this point and
        // component. Compile only that actual PV moment, retaining its full
        // recurrence DAG and batched scalar masters, to keep this regression
        // smaller than the separate complete rank-six qualification run.
        let points = [
            [0i64, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ];
        let masses = [4i64, 9, 16, 25, 36];
        let mut settings = ReductionSettings::new(5, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let mut compiler = PvCompiler::with_aliases(
            points.iter().map(|point| point.map(Atom::num)).collect(),
            masses.map(Atom::num).to_vec(),
        )
        .unwrap();
        let monomial = Monomial([5, 0, 0, 1]);
        let moment = compiler.compile_moment(monomial.0).unwrap();
        plan.settings.rank = 6;
        plan.basis = vec![monomial];
        plan.moments = vec![convert_convention(&moment, monomial, &plan.settings)];
        plan.aliases = compiler.aliases().to_vec();
        plan.leaves = compiler.leaves().to_vec();
        plan.master_arguments = plan
            .leaves
            .iter()
            .map(|leaf| {
                let pairs: &[(usize, usize)] = match leaf.active.len() {
                    1 => &[],
                    2 => &[(0, 1)],
                    3 => &[(0, 1), (1, 2), (0, 2)],
                    4 => &[(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)],
                    _ => unreachable!(),
                };
                let mut arguments: Vec<_> = pairs
                    .iter()
                    .map(|&(i, j)| {
                        let difference = std::array::from_fn::<_, 4, _>(|mu| {
                            points[leaf.active[i]][mu] - points[leaf.active[j]][mu]
                        });
                        Atom::num(
                            difference[0] * difference[0]
                                - difference[1] * difference[1]
                                - difference[2] * difference[2]
                                - difference[3] * difference[3],
                        )
                    })
                    .collect();
                arguments.extend(leaf.active.iter().map(|&i| Atom::num(masses[i])));
                arguments.push(Atom::num(9));
                arguments
            })
            .collect();
        // The fixture already compiles its exact zero-invariant branches.
        plan.special_kinematics = true;
        let values: Vec<_> = points
            .into_iter()
            .flatten()
            .chain(masses)
            .chain([3])
            .map(|value| complex(value as f64))
            .collect();
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        };
        let mut workspace =
            PrecisionRescueWorkspace::new(&plan, 0.0, RescueSettings::default()).unwrap();
        let result = workspace.reduce_batch(&batch).unwrap();
        let LaneStatus::Unstable(reason) = &workspace.last_primary_statuses()[0] else {
            panic!("the pentagon primary must be rejected by its stability estimate")
        };
        assert!(
            reason.contains("compounded master cancellation"),
            "{reason}"
        );
        assert_eq!(workspace.last_execution().requested_rows, 1);
        assert_eq!(
            workspace.last_execution().accepted_rows,
            1,
            "rescue statuses={:?}; final statuses={:?}",
            workspace.last_rescue_statuses(),
            result.statuses,
        );
        assert!(matches!(result.statuses[0], LaneStatus::Success));
        let finite = result.lane(0).unwrap()[3];
        let expected = -2.43837893544712293939249874984327534762e-3;
        assert!(finite.re.abs() < 1e-13);
        assert!((finite.im - expected).abs() < 1e-13, "{finite:?}");
    }

    #[test]
    fn finite_cancellation_fails_but_small_absolute_zero_passes() {
        let monitor = synthetic(0, false);
        let input = [complex(1.0)];
        assert!(
            monitor
                .inspect_lane(
                    &[complex(1.0), complex(1e12), complex(-1e12)],
                    &input,
                    1,
                    &0.0
                )
                .is_some()
        );
        assert!(
            monitor
                .inspect_lane(
                    &[complex(0.0), complex(2.0), complex(-2.0)],
                    &input,
                    1,
                    &0.0
                )
                .is_none()
        );
    }

    #[test]
    fn denominator_zero_is_distinct_from_a_zero_tensor_component() {
        let monitor = synthetic(2, true);
        let result = monitor
            .inspect_lane(
                &[complex(0.0), complex(2.0), complex(-2.0)],
                &[complex(1.0)],
                1,
                &0.0,
            )
            .unwrap();
        assert!(result.contains("zero recurrence denominator"));
        assert!(
            monitor
                .inspect_lane(
                    &[complex(1e-13), complex(1.0), complex(-1.0)],
                    &[complex(1.0)],
                    1,
                    &0.0
                )
                .is_some()
        );
    }

    #[test]
    fn dimensional_normalization_and_taylor_axis_are_preserved() {
        let monitor = synthetic(2, false);
        let values = [
            complex(0.0),
            complex(3.0),
            complex(2e12),
            complex(1e30),
            complex(-2e12),
            complex(-1e30),
        ];
        let input = [complex(1e6), complex(0.0)];
        let result = monitor.inspect_lane(&values, &input, 2, &0.0).unwrap();
        assert!(result.contains("Taylor component 1"));
        assert!(
            monitor
                .inspect_lane(
                    &[complex(0.0), complex(2e12), complex(-2e12)],
                    &[complex(1e6)],
                    1,
                    &0.0
                )
                .is_none()
        );
    }

    #[test]
    fn precision_threshold_survives_beyond_binary64_exponent_range() {
        let prototype = Float::with_val(1536, 0);
        let tolerance = dyadic(&prototype, -1517);
        assert!(!tolerance.is_zero());
        let monitor = synthetic(0, false);
        let one = prototype.one();
        assert!(
            monitor
                .inspect_lane(
                    &[
                        complex(prototype.zero()),
                        complex(one.clone()),
                        complex(-one.clone())
                    ],
                    &[complex(one)],
                    1,
                    &prototype
                )
                .is_none()
        );
    }

    #[test]
    fn actual_finite_small_gram_pv_bubble_is_flagged_lane_locally() {
        use crate::{Algorithm, KinematicsBatch, LaneStatus, ReductionSettings, Workspace};

        let mut settings = ReductionSettings::new(2, 1);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let plan = ReductionPlan::new(settings).unwrap();
        let mut workspace = Workspace::new(&plan, 0.0).unwrap();
        let mut values = Vec::new();
        for spatial in [1.0, 1e-5] {
            values.extend(
                [0.0, 0.0, 0.0, 0.0, 0.0, spatial, 0.0, 0.0, 4.0, 9.0, 3.0]
                    .into_iter()
                    .map(complex),
            );
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values,
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        match &result.statuses[1] {
            LaneStatus::Unstable(message) => {
                assert!(message.contains("estimated cancellation"), "{message}")
            }
            status => panic!("finite small-Gram reduction was not diagnosed: {status:?}"),
        }
        assert!(
            result
                .lane(0)
                .unwrap()
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    #[test]
    fn conditional_alias_monitors_preserve_inactive_poles_and_union_paths() {
        use crate::{Algorithm, KinematicsBatch, LaneStatus, ReductionSettings, Workspace};

        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let x = plan.parameters[0].clone();
        let y = plan.parameters[1].clone();
        let node = Atom::var(symbolica::symbol!("iregi::stability_test_singular_alias"));
        plan.aliases.push((node.clone(), Atom::num(1) + x.pow(-1)));
        plan.moments = vec![[
            Symbol::IF.call((&x, &node, Atom::num(2))),
            Symbol::IF.call((&y, Atom::new(), &node)),
            Atom::new(),
            Atom::new(),
        ]];
        let mut workspace = Workspace::new(&plan, 0.0).unwrap();
        let mut values = Vec::new();
        for (x, y) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)] {
            values.extend([x, y, 0.0, 0.0, 4.0, 3.0].into_iter().map(complex));
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 1,
                values,
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert_eq!(result.statuses[1], LaneStatus::Success);
        assert_eq!(result.lane(0).unwrap()[0], complex(2.0));
        assert_eq!(result.lane(1).unwrap()[0], complex(2.0));
        assert!(matches!(result.statuses[2], LaneStatus::Unstable(_)));
    }

    #[test]
    fn normalization_range_loss_is_reported() {
        let overflow = synthetic(4, false);
        let underflow = synthetic(2, false);
        let values = [complex(0.0), complex(0.0), complex(0.0)];
        assert!(
            overflow
                .inspect_lane(&values, &[complex(1e100)], 1, &0.0)
                .unwrap()
                .contains("outside the numeric range")
        );
        assert!(
            underflow
                .inspect_lane(&values, &[complex(1e-200)], 1, &0.0)
                .unwrap()
                .contains("outside the numeric range")
        );
    }

    #[test]
    fn promoted_arithmetic_retains_the_requested_accuracy() {
        let prototype = Float::with_val(256, 0);
        let huge = prototype.from_rational(&Rational::from(Integer::from(10).pow(20)));
        let values = [
            complex(prototype.one()),
            complex(huge.clone()),
            complex(-huge),
        ];
        let input = [complex(prototype.one())];
        let monitor = synthetic(0, false);
        assert!(
            monitor
                .inspect_lane(&values, &input, 1, &prototype)
                .is_some()
        );
        assert!(
            monitor
                .inspect_lane_with_accuracy(&values, &input, 1, &prototype, 53)
                .is_none()
        );
    }

    #[test]
    fn roundoff_weights_each_component_and_ignores_zero_metadata() {
        let prototype = Float::new(256);
        let mut roundoff = OperandRoundoff::new(&prototype);
        let strong = Float::with_val(256, 1);
        let zero_with_one_bit = Float::new(1);
        let reference = roundoff
            .complex(&Complex::new(strong.clone(), zero_with_one_bit))
            .unwrap();
        let tiny_weak = Float::with_val(80, 2).pow(300).inv();
        let mixed = roundoff.complex(&Complex::new(strong, tiny_weak)).unwrap();
        assert!(mixed <= reference.clone() * Float::with_val(256, 2));
        assert!(mixed < dyadic(&prototype, -240));
        assert!(
            roundoff
                .complex(&Complex::new(Float::new(1), Float::new(1)))
                .unwrap()
                .is_zero()
        );
    }

    #[test]
    fn tiny_low_relative_precision_terms_use_the_unchanged_absolute_budget() {
        let prototype = Float::new(256);
        let mut tiny = Float::with_val(256, 2).pow(200).inv();
        tiny.set_prec(1);
        let mut roundoff = OperandRoundoff::new(&prototype);
        let estimate = roundoff.complex(&complex(tiny.clone())).unwrap();
        assert!(!estimate.is_zero());
        assert!(estimate > tiny);
        // Symbolica can clamp a tiny cancellation residue to one significant
        // bit. Its relative estimate exceeds one, but its weighted absolute
        // estimate can still satisfy the requested dimensional target. The
        // precursor recurrence and positive-envelope checks remain active.
        let monitor = synthetic(0, false);
        let input = [complex(prototype.one())];
        assert!(
            monitor
                .inspect_lane_with_accuracy(
                    &[
                        complex(tiny.clone()),
                        complex(tiny),
                        complex(prototype.zero())
                    ],
                    &input,
                    1,
                    &prototype,
                    128,
                )
                .is_none()
        );
        let material = Float::with_val(1, 1);
        assert!(
            monitor
                .inspect_lane_with_accuracy(
                    &[
                        complex(material.clone()),
                        complex(material),
                        complex(prototype.zero())
                    ],
                    &input,
                    1,
                    &prototype,
                    128,
                )
                .is_some()
        );
    }

    #[test]
    fn weighted_roundoff_keeps_underflow_visible_without_rejecting_tiny_absolute_errors() {
        let mut roundoff = OperandRoundoff::new(&0.0);
        let tiny = f64::from_bits(1);
        let estimate = roundoff.complex(&complex(tiny)).unwrap();
        assert_eq!(estimate, tiny);
        let mut monitor = synthetic(0, false);
        monitor.output_dimensions = vec![0];
        assert!(
            monitor
                .inspect_output_precision_lane(&[complex(tiny)], &[complex(1.0)], 1, &0.0, 53)
                .is_none()
        );
        monitor.output_dimensions = vec![2];
        assert!(
            monitor
                .inspect_output_precision_lane(
                    &[complex(tiny)],
                    &[complex(tiny.sqrt())],
                    1,
                    &0.0,
                    53
                )
                .is_some()
        );
        // DoubleFloat's high limb is normal here, but its low limb has no
        // exponent headroom for the nominal106-bit precision. It must use
        // the full range check, rather than the binary64 normal fast path.
        let tiny_dd = iregi_masters::DoubleFloat::from(f64::MIN_POSITIVE);
        let zero_dd = iregi_masters::DoubleFloat::from(0.0);
        assert!(
            monitor
                .inspect_output_precision_lane(
                    &[complex(tiny_dd)],
                    &[complex(tiny_dd.sqrt())],
                    1,
                    &zero_dd,
                    106
                )
                .is_some()
        );
    }

    #[test]
    fn master_precision_provenance_retains_the_error_budget_and_checks_full_jets() {
        let prototype = Float::new(128);
        let c = |bits, value| Complex::new(Float::with_val(bits, value), Float::new(bits));
        assert!(
            master_argument_precision_failure(&[c(125, 2), c(125, 1)], &[], 1, &prototype, 128)
                .is_none()
        );
        assert!(
            master_argument_precision_failure(&[c(80, 2), c(80, 1)], &[], 1, &prototype, 128)
                .is_some()
        );
        // The scale is irrelevant only for an entire constant scaleless jet.
        assert!(
            master_argument_precision_failure(
                &[c(1, 0), c(1, 0), c(80, 1), c(1, 0)],
                &[],
                2,
                &prototype,
                128
            )
            .is_none()
        );
        assert!(
            master_argument_precision_failure(
                &[c(128, 0), c(80, 3), c(128, 1), c(128, 0)],
                &[],
                2,
                &prototype,
                128
            )
            .is_some()
        );
    }

    #[test]
    fn inactive_master_requests_are_skipped_per_sample() {
        use crate::{Algorithm, KinematicsBatch, LaneStatus, ReductionSettings, Workspace};
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let x = &plan.parameters[0];
        plan.moments = vec![[
            Symbol::IF.call((x, &plan.leaves[0].slots[0], Atom::new())),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ]];
        let mut workspace = Workspace::new(&plan, 0.0).unwrap();
        let values = [0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 3.0]
            .into_iter()
            .map(complex)
            .collect();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values,
            })
            .unwrap();
        assert_eq!(result.statuses, vec![LaneStatus::Success; 2]);
        assert_eq!(result.lane(0).unwrap()[0], complex(0.0));
        assert_eq!(result.lane(1).unwrap()[0], complex(4.0));
        assert_eq!(workspace.last_execution().master_rows, 1);
    }

    #[test]
    fn master_dependent_activation_is_conservative_and_not_circular() {
        use crate::{Algorithm, ReductionSettings};
        let mut settings = ReductionSettings::new(2, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let slots = std::array::from_fn(|i| {
            Atom::var(symbolica::symbol!(format!(
                "iregi::activation_test_master_slot_{i}"
            )))
        });
        let condition = Atom::var(symbolica::symbol!(
            "iregi::activation_test_master_condition"
        ));
        plan.aliases
            .push((condition.clone(), &slots[0] + &plan.parameters[0]));
        plan.moments = vec![[
            Symbol::IF.call((&condition, &plan.leaves[0].slots[3], Atom::new())),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ]];
        plan.leaves.push(crate::ibp::MasterLeaf {
            kind: crate::ibp::MasterLeafKind::FourDimensional,
            active: vec![0],
            slots,
        });
        plan.master_arguments.push(vec![
            plan.parameters[plan.layout.masses.start].clone(),
            plan.parameters[plan.layout.scale].pow(2),
        ]);
        let monitor = StabilityPlan::prepare(&plan).unwrap();
        assert_eq!(monitor.master_activations(), [Atom::num(1), Atom::num(1)]);
        let masters: BTreeSet<_> = plan
            .leaves
            .iter()
            .flat_map(|leaf| leaf.slots.iter().cloned())
            .collect();
        for (_, body) in monitor.master_activation_aliases() {
            assert!(
                body.get_all_symbols(false)
                    .into_iter()
                    .all(|symbol| !masters.contains(&Atom::var(symbol)))
            );
        }
    }
    include!("stability/context_tests.rs");
}
