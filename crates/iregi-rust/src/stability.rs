//! Precision-aware cancellation diagnostics for the compiled recurrence graph.
//!
//! These are conservative floating-point *estimates*, not certified error
//! bounds. Symbolica expands only individual, bounded recurrence bodies and
//! evaluates all monitor expressions alongside the ordinary batch. In
//! particular, an alias graph is never expanded recursively. This preserves
//! the shared recurrence DAG and lets analytic exceptional-kinematics plans
//! be checked according to the denominators they actually use.

use std::collections::{BTreeSet, HashMap};

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::{float::Complex, integer::Integer, rational::Rational},
};

use crate::{
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

/// Immutable monitor expressions for one reduction graph.
///
/// The executor appends `expressions()` to its ordinary expression roots and
/// evaluates them using the same aliases, numeric domain, and Dualizer.
#[derive(Clone, Debug)]
pub(crate) struct StabilityPlan {
    expressions: Vec<Atom>,
    predicate_aliases: Vec<(Atom, Atom)>,
    master_activations: Vec<Atom>,
    master_activation_aliases: Vec<(Atom, Atom)>,
    checks: Vec<Check>,
    layout: InputLayout,
}

impl StabilityPlan {
    pub(crate) fn prepare(plan: &ReductionPlan) -> Result<Self, ReductionError> {
        let mut monitor = Self {
            expressions: Vec::new(),
            predicate_aliases: Vec::new(),
            master_activations: Vec::new(),
            master_activation_aliases: Vec::new(),
            checks: Vec::new(),
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
            for slot in &leaf.slots {
                dimensions.insert(slot.clone(), 4 - 2 * leaf.active.len() as i32);
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
        monitor.prepare_master_activations(plan);
        Ok(monitor)
    }

    pub(crate) fn expressions(&self) -> &[Atom] {
        &self.expressions
    }

    pub(crate) fn aliases(&self) -> &[(Atom, Atom)] {
        &self.predicate_aliases
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
        for expression in plan.moments.iter().flatten().chain(&self.expressions) {
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
        if components == 0 || values.len() != self.expressions.len() * components {
            return Some("invalid stability-monitor buffer".into());
        }
        let precision = prototype.get_precision();
        let unit = dyadic(prototype, 1i64 - i64::from(precision));
        // At binary64 this is 2^-34, below the 1e-10 validation target.
        // Additional precision tightens the criterion without an f64 detour.
        let tolerance = dyadic(prototype, 19i64 - i64::from(accuracy_bits));
        let scale = self.kinematic_scale(input, components, prototype);
        for check in &self.checks {
            let count = if check.denominator { 1 } else { components };
            let roundoff = unit.clone() * prototype.from_usize(check.operations.saturating_mul(8));
            for component in 0..count {
                let value = &values[check.value * components + component];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Some(format!("nonfinite stability monitor for {}", check.label));
                }
                let magnitude = NativeFloat::hypot(&value.re, &value.im);
                if check.denominator && magnitude.is_zero() {
                    return Some(format!("zero recurrence denominator: {}", check.label));
                }
                let mut contribution_scale = prototype.zero();
                for &index in &check.terms {
                    let term = &values[index * components + component];
                    if !term.re.is_finite() || !term.im.is_finite() {
                        return Some(format!(
                            "nonfinite stability contribution for {}",
                            check.label
                        ));
                    }
                    contribution_scale += NativeFloat::hypot(&term.re, &term.im);
                }
                let reference = if check.denominator {
                    magnitude
                } else {
                    let natural = signed_power(&scale, check.dimension);
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
                let estimate = roundoff.clone() * contribution_scale;
                let target = tolerance.clone() * &reference;
                if !target.is_finite() || (target.is_zero() && !reference.is_zero()) {
                    return Some(format!(
                        "conditioning tolerance is outside the numeric range for {}",
                        check.label
                    ));
                }
                if !estimate.is_finite() || estimate > target {
                    return Some(format!(
                        "estimated cancellation exceeds 2^(19-{accuracy_bits}) at working precision {precision} in {}, Taylor component {component}; this estimate is not a certified error bound",
                        check.label,
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
    if expression == Atom::num(1) || expression.is_zero() {
        return expression;
    }
    if let Some(alias) = indices.get(&expression) {
        return alias.clone();
    }
    let alias = Atom::var(symbolica::symbol!(format!(
        "iregi::master_active_path_{}",
        aliases.len()
    )));
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
    use symbolica::domains::float::{Float, FloatLike, SingleFloat};

    fn synthetic(dimension: i32, denominator: bool) -> StabilityPlan {
        StabilityPlan {
            expressions: vec![Atom::new(); 3],
            predicate_aliases: Vec::new(),
            master_activations: Vec::new(),
            master_activation_aliases: Vec::new(),
            checks: vec![Check {
                value: 0,
                terms: vec![1, 2],
                dimension,
                operations: 3,
                denominator,
                label: "synthetic cancellation".into(),
            }],
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
}
