//! Workspace constants and lane-local dimensional normalization for guards.
use super::{OperandRoundoff, StabilityPlan, dyadic, signed_power};
use iregi_masters::NativeFloat;
use std::collections::{BTreeSet, HashMap};
use symbolica::domains::float::Complex;

/// Numeric state belongs to one workspace, never the shared exact plan.
#[derive(Clone)]
pub(crate) struct GuardContext<T> {
    pub(super) tolerance: T,
    pub(super) prototype_unit: T,
    dimensions: Vec<i32>,
    pub(super) local_slots: Vec<usize>,
    pub(super) primal_slots: Vec<usize>,
    pub(super) output_slots: Vec<usize>,
    scale: Option<T>,
    natural: Vec<Option<T>>,
}

impl<T: NativeFloat> GuardContext<T> {
    pub(crate) fn new(plan: &StabilityPlan, prototype: &T, accuracy_bits: u32) -> Self {
        let dimensions: Vec<_> = plan
            .checks
            .iter()
            .filter(|c| !c.denominator)
            .map(|c| c.dimension)
            .chain(plan.primal_checks.iter().map(|c| c.dimension))
            .chain(plan.output_dimensions.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let slots: HashMap<_, _> = dimensions
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, dimension)| (dimension, slot))
            .collect();
        let local_slots = plan
            .checks
            .iter()
            .map(|c| {
                if c.denominator {
                    usize::MAX
                } else {
                    slots[&c.dimension]
                }
            })
            .collect();
        let primal_slots = plan
            .primal_checks
            .iter()
            .map(|c| slots[&c.dimension])
            .collect();
        let output_slots = plan.output_dimensions.iter().map(|d| slots[d]).collect();
        let natural = vec![None; dimensions.len()];
        Self {
            tolerance: dyadic(prototype, 19i64 - i64::from(accuracy_bits)),
            prototype_unit: OperandRoundoff::new(prototype).prototype_unit,
            dimensions,
            local_slots,
            primal_slots,
            output_slots,
            scale: None,
            natural,
        }
    }

    pub(crate) fn set_accuracy(&mut self, prototype: &T, accuracy_bits: u32) {
        self.tolerance = dyadic(prototype, 19i64 - i64::from(accuracy_bits));
    }

    /// Reset unconditionally, even when the next lane has equal numeric values.
    pub(crate) fn begin_lane(&mut self) {
        self.scale = None;
        self.natural.fill(None);
    }

    pub(super) fn natural(
        &mut self,
        slot: usize,
        plan: &StabilityPlan,
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
    ) -> T {
        if let Some(value) = &self.natural[slot] {
            return value.clone();
        }
        let scale = self
            .scale
            .get_or_insert_with(|| plan.kinematic_scale(input, components, prototype));
        // Keep pow(0) at scale precision; negative powers remain pow(abs).inv().
        let value = signed_power(scale, self.dimensions[slot]);
        self.natural[slot] = Some(value.clone());
        value
    }
}
