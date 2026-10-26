//! Denominator-only transport for the positive-power affine reduction DAG.
//!
//! This extends the existing `thin_n6.rs` and source-only `pinched_batch.rs`
//! gathering pattern. The DAG owns terminal interning and the exact demanded
//! terminal profiles. This module performs no algebra and creates no values
//! for an undemanded or failed terminal. Numerator q, rank, h, scalar scale,
//! requested algorithm and every ordered user component remain unchanged.

use std::collections::{BTreeMap, BTreeSet};

use iregi_masters::NativeFloat;

use super::dag::Terminal;
use crate::{InputLayout, InputPolicy, InvariantInput, KinematicsBatch, ReductionError, ReductionSettings};

fn invalid(message: &str) -> ReductionError {
    ReductionError::InvalidInput(message.into())
}

fn product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("affine-power packed dimensions overflow"))
}

fn parameters(n: usize) -> Result<usize, ReductionError> {
    product(n, 5)?
        .checked_add(1)
        .ok_or_else(|| invalid("affine-power parameter count overflows"))
}

fn input_layout(n: usize, invariants: InvariantInput) -> Result<InputLayout, ReductionError> {
    let shifts = 0..product(n, 4)?;
    let products = if invariants == InvariantInput::SuppliedScalarProducts {
        let next = n.checked_add(1)
            .ok_or_else(|| invalid("affine-power triangular parameter count overflows"))?;
        let count = product(n, next)? / 2;
        let end = shifts.end.checked_add(count)
            .ok_or_else(|| invalid("affine-power scalar-product layout overflows"))?;
        Some(shifts.end..end)
    } else { None };
    let mass_start = products.as_ref().map_or(shifts.end, |range| range.end);
    let scale = mass_start.checked_add(n)
        .ok_or_else(|| invalid("affine-power mass layout overflows"))?;
    // Retain the original Cartesian parameter-count overflow check.
    if products.is_none() { parameters(n)?; }
    Ok(InputLayout { shifts, scalar_products: products, masses: mass_start..scale, scale })
}

/// Argument positions only. The canonical child upper triangle can reverse
/// original line order, so deletion alone is not a valid G gathering map.
fn product_arguments(
    layout: &InputLayout,
    n: usize,
    original_lines: &[usize],
) -> Result<Vec<usize>, ReductionError> {
    let Some(products) = &layout.scalar_products else { return Ok(Vec::new()); };
    let mut row_starts = Vec::new();
    reserve(&mut row_starts, n)?;
    let mut argument = products.start;
    for row in 0..n {
        row_starts.push(argument);
        argument = argument.checked_add(n - row)
            .ok_or_else(|| invalid("affine-power scalar-product row offset overflows"))?;
    }
    if argument != products.end {
        return Err(invalid("affine-power scalar-product triangle differs from layout"));
    }
    let mut arguments = Vec::new();
    let next = original_lines.len().checked_add(1)
        .ok_or_else(|| invalid("affine-power canonical product count overflows"))?;
    reserve(&mut arguments, product(original_lines.len(), next)? / 2)?;
    for (position, &left) in original_lines.iter().enumerate() {
        for &right in &original_lines[position..] {
            let first = left.min(right);
            let second = left.max(right);
            let start = *row_starts.get(first)
                .ok_or_else(|| invalid("affine-power canonical product line is out of range"))?;
            if second >= n {
                return Err(invalid("affine-power canonical product line is out of range"));
            }
            arguments.push(start.checked_add(second - first)
                .ok_or_else(|| invalid("affine-power canonical product argument overflows"))?);
        }
    }
    Ok(arguments)
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), ReductionError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| invalid("affine-power packed allocation exceeds available resources"))
}

#[derive(Clone, Debug)]
struct TerminalLayout {
    group: usize,
    /// Original parent denominator indices in canonical child order.
    original_lines: Vec<usize>,
    /// Original parent G arguments in the canonical child's upper-triangle order.
    product_arguments: Vec<usize>,
}

/// Structural child groups. No point-count, rank or h restriction is imposed
/// by this transport helper. Ordinary construction requires Preserve and
/// FromShifts. The separately named private supplied-G transport is usable only
/// behind a fresh runtime certificate; structural construction is not a proof.
#[derive(Clone, Debug)]
pub(super) struct PowerBatchLayout {
    parent_layout: InputLayout,
    parent_parameters: usize,
    child_parameters: usize,
    child_settings: Vec<ReductionSettings>,
    terminals: Vec<TerminalLayout>,
}

impl PowerBatchLayout {
    pub(super) fn new(
        settings: &ReductionSettings,
        terminals: &[Terminal],
    ) -> Result<Self, ReductionError> {
        Self::prepare(settings, terminals, InvariantInput::FromShifts)
    }

    /// Private gathering policy for the checked supplied-G runtime. This only
    /// validates transport metadata: it neither certifies G/p compatibility nor
    /// admits an unchecked public reduction domain. The caller must establish
    /// every required full-jet identity before any gathered row is evaluated.
    pub(super) fn new_checked_supplied(
        settings: &ReductionSettings,
        terminals: &[Terminal],
    ) -> Result<Self, ReductionError> {
        Self::prepare(settings, terminals, InvariantInput::SuppliedScalarProducts)
    }

    fn prepare(
        settings: &ReductionSettings,
        terminals: &[Terminal],
        invariants: InvariantInput,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if n < 2
            || settings.powers.len() != n
            || settings.powers.contains(&0)
            || settings.invariants != invariants
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(invalid(if invariants == InvariantInput::FromShifts {
                "affine-power child packing requires N>=2, N positive powers, FromShifts and Preserve"
            } else {
                "checked affine-power supplied-G packing requires N>=2, N positive powers, SuppliedScalarProducts and Preserve; construction provides no compatibility proof"
            }));
        }
        let parent_layout = input_layout(n, invariants)?;
        let parent_parameters = parent_layout.scale.checked_add(1)
            .ok_or_else(|| invalid("affine-power parent parameter count overflows"))?;
        let child_parameters = input_layout(n - 1, invariants)?.scale.checked_add(1)
            .ok_or_else(|| invalid("affine-power child parameter count overflows"))?;
        let mut groups = BTreeMap::<Vec<u16>, usize>::new();
        let mut child_settings = Vec::new();
        let mut layouts = Vec::new();
        reserve(&mut layouts, terminals.len())?;
        for terminal in terminals {
            if terminal.removed_line >= n
                || terminal.remaining_powers.len() != n - 1
                || terminal.remaining_powers.contains(&0)
                || terminal.canonical_to_remaining.len() != n - 1
            {
                return Err(invalid(
                    "affine-power terminal has invalid removed line or powers",
                ));
            }
            let mut expected: Vec<_> = (0..n - 1).collect();
            // Metadata ordering only. Remaining positions preserve original
            // line order, so their numeric order is the stable tie breaker.
            expected.sort_by_key(|&i| (std::cmp::Reverse(terminal.remaining_powers[i]), i));
            if terminal.canonical_to_remaining != expected {
                return Err(invalid(
                    "affine-power terminal permutation is not stable descending-power order",
                ));
            }
            for (position, &power) in terminal.remaining_powers.iter().enumerate() {
                let original = position + usize::from(position >= terminal.removed_line);
                if power > settings.powers[original] {
                    return Err(invalid(
                        "affine-power terminal raises a parent propagator power",
                    ));
                }
            }
            let powers = terminal.canonical_powers();
            let original_lines = terminal.canonical_original_lines();
            let product_arguments = product_arguments(&parent_layout, n, &original_lines)?;
            let group = if let Some(&group) = groups.get(&powers) {
                group
            } else {
                let group = child_settings.len();
                let mut child = settings.clone();
                child.propagators = n - 1;
                child.powers = powers.clone();
                // The identity changes denominators only. Keep the complete
                // same-q tensor basis, rank and h even for repeated powers.
                child_settings.push(child);
                groups.insert(powers, group);
                group
            };
            layouts.push(TerminalLayout {
                group,
                original_lines,
                product_arguments,
            });
        }
        Ok(Self {
            parent_layout,
            parent_parameters,
            child_parameters,
            child_settings,
            terminals: layouts,
        })
    }

    /// One template per distinct canonical child-power vector, in first
    /// terminal occurrence order. The requested algorithm is never rewritten.
    pub(super) fn child_settings(&self) -> &[ReductionSettings] {
        &self.child_settings
    }

    pub(super) fn terminal_count(&self) -> usize {
        self.terminals.len()
    }

    /// Gather exactly the terminal IDs demanded for each physical parent row.
    /// Repeated IDs within a row are malformed; equivalent DAG paths must have
    /// been interned before this call. Component slots remain argument-major,
    /// inside their own physical rows; they never become extra sample lanes.
    ///
    /// A group with zero rows must not be submitted to a child evaluator.
    /// Absence in the lookup denotes no demand, never a successful zero.
    pub(super) fn pack<T: NativeFloat>(
        &self,
        parent: &KinematicsBatch<T>,
        demanded: &[Vec<usize>],
        max_packed_rows: usize,
    ) -> Result<PackedPowerChildren<T>, ReductionError> {
        if parent.components == 0 || demanded.len() != parent.rows {
            return Err(invalid(
                "affine-power parent rows/components/demand profiles differ",
            ));
        }
        let parent_width = product(self.parent_parameters, parent.components)?;
        if parent.values.len() != product(parent.rows, parent_width)? {
            return Err(invalid("affine-power parent input dimensions differ"));
        }
        let child_width = product(self.child_parameters, parent.components)?;
        let lookup_len = product(parent.rows, self.terminals.len())?;
        let mut group_rows = vec![0usize; self.child_settings.len()];
        let mut total_rows = 0usize;
        for profile in demanded {
            let mut seen = BTreeSet::new();
            for &terminal in profile {
                let Some(layout) = self.terminals.get(terminal) else {
                    return Err(invalid(
                        "affine-power demand has an out-of-range terminal ID",
                    ));
                };
                if !seen.insert(terminal) {
                    return Err(invalid("affine-power demand repeats a terminal ID"));
                }
                total_rows = total_rows
                    .checked_add(1)
                    .ok_or_else(|| invalid("affine-power packed row count overflows"))?;
                group_rows[layout.group] = group_rows[layout.group]
                    .checked_add(1)
                    .ok_or_else(|| invalid("affine-power group row count overflows"))?;
            }
        }
        if total_rows > max_packed_rows {
            return Err(invalid(
                "affine-power packed child row resource budget exceeded",
            ));
        }
        // Check every size before copying any input or constructing output.
        product(total_rows, child_width)?;
        for &rows in &group_rows {
            product(rows, child_width)?;
        }
        let mut lookup = Vec::new();
        reserve(&mut lookup, lookup_len)?;
        lookup.resize(lookup_len, None);
        let mut groups = Vec::new();
        reserve(&mut groups, group_rows.len())?;
        for &rows in &group_rows {
            let mut values = Vec::new();
            reserve(&mut values, product(rows, child_width)?)?;
            let mut origins = Vec::new();
            reserve(&mut origins, rows)?;
            groups.push(PackedPowerGroup {
                input: KinematicsBatch {
                    rows,
                    components: parent.components,
                    values,
                },
                origins,
            });
        }
        // Terminal-major order keeps all physical rows of each exact terminal
        // adjacent within its canonical-power group. Demand-list order cannot
        // alter structural grouping or the inverse mapping.
        for (terminal_index, terminal) in self.terminals.iter().enumerate() {
            for (parent_row, profile) in demanded.iter().enumerate() {
                if !profile.contains(&terminal_index) {
                    continue;
                }
                let source =
                    &parent.values[parent_row * parent_width..(parent_row + 1) * parent_width];
                let group = &mut groups[terminal.group];
                let child_row = group.origins.len();
                let location = PackedChildRow {
                    group: terminal.group,
                    child_row,
                };
                lookup[parent_row * self.terminals.len() + terminal_index] = Some(location);
                group.origins.push(PowerChildOrigin {
                    parent_row,
                    terminal_index,
                });
                let mut copy = |parameter: usize| {
                    group.input.values.extend_from_slice(
                        &source[parameter * parent.components..(parameter + 1) * parent.components],
                    );
                };
                for &line in &terminal.original_lines {
                    for coordinate in 0..4 {
                        copy(4 * line + coordinate);
                    }
                }
                for &argument in &terminal.product_arguments {
                    copy(argument);
                }
                for &line in &terminal.original_lines {
                    copy(self.parent_layout.masses.start + line);
                }
                copy(self.parent_layout.scale);
            }
        }
        for group in &groups {
            debug_assert_eq!(group.origins.len(), group.input.rows);
            debug_assert_eq!(group.input.values.len(), group.input.rows * child_width);
        }
        Ok(PackedPowerChildren {
            groups,
            parent_rows: parent.rows,
            terminal_count: self.terminals.len(),
            lookup,
            total_rows,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedChildRow {
    pub group: usize,
    pub child_row: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PowerChildOrigin {
    pub parent_row: usize,
    pub terminal_index: usize,
}

pub(super) struct PackedPowerGroup<T: NativeFloat> {
    pub input: KinematicsBatch<T>,
    /// This inverse mapping also identifies failed child lanes. The owner
    /// must preserve their failure rather than substitute a tensor zero.
    pub origins: Vec<PowerChildOrigin>,
}

pub(super) struct PackedPowerChildren<T: NativeFloat> {
    pub groups: Vec<PackedPowerGroup<T>>,
    pub total_rows: usize,
    parent_rows: usize,
    terminal_count: usize,
    lookup: Vec<Option<PackedChildRow>>,
}

impl<T: NativeFloat> PackedPowerChildren<T> {
    pub(super) fn location(
        &self,
        parent_row: usize,
        terminal_index: usize,
    ) -> Option<PackedChildRow> {
        if parent_row >= self.parent_rows || terminal_index >= self.terminal_count {
            return None;
        }
        self.lookup[parent_row * self.terminal_count + terminal_index]
    }

    /// Fail closed when an executor asks for a terminal it did not demand.
    /// There is no numerical zero, default child row or implicit profile.
    pub(super) fn require_location(
        &self,
        parent_row: usize,
        terminal_index: usize,
    ) -> Result<PackedChildRow, ReductionError> {
        self.location(parent_row, terminal_index)
            .ok_or_else(|| invalid("affine-power requested terminal has no packed child row"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, Convention, TensorWeights};
    use symbolica::domains::float::Complex;

    fn terminal(removed_line: usize, powers: &[u16], permutation: &[usize]) -> Terminal {
        Terminal {
            removed_line,
            remaining_powers: powers.to_vec(),
            canonical_to_remaining: permutation.to_vec(),
        }
    }

    fn input(n: usize, rows: usize, components: usize) -> KinematicsBatch<f64> {
        KinematicsBatch {
            rows,
            components,
            values: (0..rows * (5 * n + 1) * components)
                .map(|index| Complex::new(index as f64 + 1., -(index as f64) - 3.))
                .collect(),
        }
    }

    #[test]
    fn canonical_power_groups_keep_full_rank_same_h_and_requested_semantics() {
        for (n, h) in [(6, -1), (6, 1), (17, 4)] {
            let mut parent = ReductionSettings::new(n, 6);
            parent.powers = vec![3; n];
            parent.dimension_shift = h;
            parent.algorithm = Algorithm::Ibp;
            parent.convention = Convention::OneLoop;
            parent.tensor_weights = TensorWeights::Multinomial;
            let mut a = vec![1; n - 1];
            a[1] = 2;
            let mut a_order = vec![1, 0];
            a_order.extend(2..n - 1);
            let mut b = vec![1; n - 1];
            b[0] = 2;
            let unit = vec![1; n - 1];
            let identity: Vec<_> = (0..n - 1).collect();
            let terminals = [
                terminal(0, &a, &a_order),
                terminal(n - 1, &b, &identity),
                terminal(1, &unit, &identity),
            ];
            let layout = PowerBatchLayout::new(&parent, &terminals).unwrap();
            assert_eq!(layout.terminal_count(), 3);
            assert_eq!(layout.child_settings().len(), 2);
            for child in layout.child_settings() {
                assert_eq!(child.propagators, n - 1);
                assert_eq!(child.rank, parent.rank);
                assert_eq!(child.dimension_shift, parent.dimension_shift);
                assert_eq!(child.algorithm, parent.algorithm);
                assert_eq!(child.convention, parent.convention);
                assert_eq!(child.tensor_weights, parent.tensor_weights);
                assert_eq!(child.invariants, InvariantInput::FromShifts);
                assert_eq!(child.input_policy, InputPolicy::Preserve);
            }
            assert_eq!(layout.child_settings()[0].powers, b);
            assert_eq!(layout.child_settings()[1].powers, unit);
        }
    }

    #[test]
    fn odd_physical_rows_all_complex_components_permutations_and_mu_survive() {
        let mut settings = ReductionSettings::new(4, 6);
        settings.powers = vec![3; 4];
        let terminals = [
            terminal(0, &[1, 3, 2], &[1, 2, 0]),
            terminal(2, &[3, 2, 1], &[0, 1, 2]),
            terminal(1, &[1, 1, 1], &[0, 1, 2]),
        ];
        let layout = PowerBatchLayout::new(&settings, &terminals).unwrap();
        let source = input(4, 3, 4);
        let packed = layout
            .pack(&source, &[vec![2, 1], vec![0], vec![1, 0]], 5)
            .unwrap();
        assert_eq!(packed.total_rows, 5);
        assert_eq!(packed.groups.len(), 2);
        assert_eq!(packed.groups[0].input.rows, 4);
        assert_eq!(packed.groups[1].input.rows, 1);
        assert_eq!(
            packed.groups[0].origins,
            [
                PowerChildOrigin {
                    parent_row: 1,
                    terminal_index: 0
                },
                PowerChildOrigin {
                    parent_row: 2,
                    terminal_index: 0
                },
                PowerChildOrigin {
                    parent_row: 0,
                    terminal_index: 1
                },
                PowerChildOrigin {
                    parent_row: 2,
                    terminal_index: 1
                },
            ]
        );
        assert_eq!(
            packed.groups[1].origins,
            [PowerChildOrigin {
                parent_row: 0,
                terminal_index: 2
            }]
        );
        let original_lines = [vec![2, 3, 1], vec![0, 1, 3], vec![0, 2, 3]];
        for (group_index, group) in packed.groups.iter().enumerate() {
            assert_eq!(group.input.components, 4);
            for (child_row, origin) in group.origins.iter().enumerate() {
                assert_eq!(
                    packed
                        .require_location(origin.parent_row, origin.terminal_index)
                        .unwrap(),
                    PackedChildRow {
                        group: group_index,
                        child_row
                    }
                );
                let parameters: Vec<_> = original_lines[origin.terminal_index]
                    .iter()
                    .flat_map(|&line| 4 * line..4 * line + 4)
                    .chain(
                        original_lines[origin.terminal_index]
                            .iter()
                            .map(|&line| 16 + line),
                    )
                    .chain(std::iter::once(20))
                    .collect();
                for (target, parameter) in parameters.into_iter().enumerate() {
                    assert_eq!(
                        &group.input.values
                            [(child_row * 16 + target) * 4..(child_row * 16 + target + 1) * 4],
                        &source.values[(origin.parent_row * 21 + parameter) * 4
                            ..(origin.parent_row * 21 + parameter + 1) * 4]
                    );
                }
            }
        }
    }

    #[test]
    fn exact_profiles_never_create_default_or_zero_rows_for_missing_terminals() {
        let mut settings = ReductionSettings::new(3, 2);
        settings.powers = vec![2; 3];
        let terminals = [terminal(0, &[2, 1], &[0, 1]), terminal(2, &[2, 1], &[0, 1])];
        let layout = PowerBatchLayout::new(&settings, &terminals).unwrap();
        let packed = layout
            .pack(&input(3, 3, 2), &[vec![], vec![1], vec![0]], 2)
            .unwrap();
        assert!(packed.location(0, 0).is_none());
        assert!(packed.require_location(0, 0).is_err());
        assert!(packed.require_location(3, 0).is_err());
        assert!(packed.require_location(0, 2).is_err());
        // A child failure at this row maps to its exact parent/terminal. It
        // cannot be substituted by another successful row in the same group.
        let failed = packed.require_location(1, 1).unwrap();
        assert_eq!(
            packed.groups[failed.group].origins[failed.child_row],
            PowerChildOrigin {
                parent_row: 1,
                terminal_index: 1
            }
        );
        assert_ne!(packed.require_location(2, 0).unwrap(), failed);
        assert_eq!(packed.total_rows, 2);
    }

    #[test]
    fn terminal_duplicates_are_not_interned_or_silently_merged_by_packing() {
        let mut settings = ReductionSettings::new(3, 2);
        settings.powers = vec![2; 3];
        let terminals = [terminal(0, &[2, 1], &[0, 1]), terminal(0, &[2, 1], &[0, 1])];
        let layout = PowerBatchLayout::new(&settings, &terminals).unwrap();
        let packed = layout.pack(&input(3, 1, 1), &[vec![0, 1]], 2).unwrap();
        assert_eq!(layout.child_settings().len(), 1);
        assert_eq!(packed.groups[0].input.rows, 2);
        assert_ne!(
            packed.require_location(0, 0).unwrap(),
            packed.require_location(0, 1).unwrap()
        );
    }

    #[test]
    fn malformed_profiles_budget_components_and_checked_dimensions_fail_before_gather() {
        let settings = ReductionSettings::new(3, 2);
        let terminals = [terminal(0, &[1, 1], &[0, 1]), terminal(2, &[1, 1], &[0, 1])];
        let layout = PowerBatchLayout::new(&settings, &terminals).unwrap();
        let source = input(3, 1, 1);
        for demands in [vec![vec![0, 0]], vec![vec![2]], vec![]] {
            assert!(layout.pack(&source, &demands, 2).is_err());
        }
        assert!(layout.pack(&source, &[vec![0, 1]], 1).is_err());
        for malformed in [
            KinematicsBatch::<f64> {
                rows: 1,
                components: 0,
                values: vec![],
            },
            KinematicsBatch::<f64> {
                rows: 1,
                components: 2,
                values: vec![],
            },
            KinematicsBatch::<f64> {
                rows: 1,
                components: usize::MAX,
                values: vec![],
            },
        ] {
            assert!(layout.pack(&malformed, &[vec![0]], 2).is_err());
        }
        assert!(product(usize::MAX, 2).is_err());
        assert!(parameters(usize::MAX).is_err());
    }

    #[test]
    fn empty_physical_batches_empty_profiles_and_empty_terminal_dags_remain_empty() {
        let settings = ReductionSettings::new(3, 2);
        let terminal = terminal(0, &[1, 1], &[0, 1]);
        let layout = PowerBatchLayout::new(&settings, &[terminal]).unwrap();
        for (source, demands) in [
            (input(3, 0, 3), vec![]),
            (input(3, 3, 3), vec![vec![], vec![], vec![]]),
        ] {
            let packed = layout.pack(&source, &demands, 0).unwrap();
            assert_eq!(packed.total_rows, 0);
            assert_eq!(packed.groups[0].input.rows, 0);
            assert!(packed.groups[0].input.values.is_empty());
            assert!(packed.groups[0].origins.is_empty());
        }
        let empty = PowerBatchLayout::new(&settings, &[]).unwrap();
        let packed = empty
            .pack(&input(3, 3, 2), &[vec![], vec![], vec![]], 0)
            .unwrap();
        assert!(packed.groups.is_empty());
        assert_eq!(packed.total_rows, 0);
        assert!(packed.require_location(0, 0).is_err());
    }

    #[test]
    fn invalid_terminal_metadata_or_initial_input_policies_are_rejected() {
        let mut settings = ReductionSettings::new(3, 2);
        settings.powers = vec![3; 3];
        for malformed in [
            terminal(3, &[1, 1], &[0, 1]),
            terminal(0, &[1], &[0]),
            terminal(0, &[0, 1], &[1, 0]),
            terminal(0, &[1, 1], &[0, 0]),
            terminal(0, &[1, 2], &[0, 1]),
            terminal(0, &[1, 1], &[1, 0]),
            terminal(0, &[4, 1], &[0, 1]),
        ] {
            assert!(PowerBatchLayout::new(&settings, &[malformed]).is_err());
        }
        let terminals = [terminal(0, &[1, 1], &[0, 1])];
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        assert!(PowerBatchLayout::new(&settings, &terminals).is_err());
        settings.invariants = InvariantInput::FromShifts;
        settings.input_policy = InputPolicy::LegacyOnShell {
            threshold: "0".into(),
        };
        assert!(PowerBatchLayout::new(&settings, &terminals).is_err());
        settings.input_policy = InputPolicy::Preserve;
        settings.powers[1] = 0;
        assert!(PowerBatchLayout::new(&settings, &terminals).is_err());
        assert!(PowerBatchLayout::new(&ReductionSettings::new(1, 2), &[]).is_err());
    }

    #[test]
    fn actual_positive_power_dag_terminals_pack_dot_triple_and_mixed_n6_profiles() {
        use super::super::dag::{AffinePowerDag, DagBudget};

        for powers in [
            vec![2, 1, 1, 1, 1, 1],
            vec![3, 1, 1, 1, 1, 1],
            vec![2, 2, 1, 1, 1, 1],
        ] {
            let dag = AffinePowerDag::new(
                powers.clone(),
                &[],
                DagBudget {
                    max_nodes: 10_000,
                    max_edges: 100_000,
                    max_terminals: 10_000,
                },
            )
            .unwrap();
            for h in [-1, 1] {
                let mut settings = ReductionSettings::new(6, 6);
                settings.powers = powers.clone();
                settings.dimension_shift = h;
                settings.algorithm = Algorithm::Ibp;
                let layout = PowerBatchLayout::new(&settings, &dag.terminals).unwrap();
                let profile: Vec<_> = (0..dag.terminals.len()).collect();
                let profiles = vec![profile.clone(), profile.clone(), profile];
                let budget = dag.terminals.len() * 3;
                let source = input(6, 3, 3);
                let packed = layout.pack(&source, &profiles, budget).unwrap();
                assert_eq!(packed.total_rows, budget);
                assert!(layout.pack(&source, &profiles, budget - 1).is_err());
                for (group_index, group) in packed.groups.iter().enumerate() {
                    let child = &layout.child_settings()[group_index];
                    assert_eq!(child.propagators, 5);
                    assert_eq!(child.rank, 6);
                    assert_eq!(child.dimension_shift, h);
                    assert_eq!(child.algorithm, Algorithm::Ibp);
                    for (child_row, origin) in group.origins.iter().enumerate() {
                        assert_eq!(
                            child.powers,
                            dag.terminals[origin.terminal_index].canonical_powers()
                        );
                        assert_eq!(
                            packed
                                .require_location(origin.parent_row, origin.terminal_index)
                                .unwrap(),
                            PackedChildRow {
                                group: group_index,
                                child_row
                            }
                        );
                    }
                }
            }
        }
    }

    fn supplied_input(n: usize, rows: usize, components: usize) -> KinematicsBatch<f64> {
        let width = input_layout(n, InvariantInput::SuppliedScalarProducts).unwrap().scale + 1;
        KinematicsBatch {
            rows,
            components,
            // Deliberately independent complex sentinels in every G slot:
            // copying them cannot be replaced by recomputing products from p.
            values: (0..rows * width * components)
                .map(|index| Complex::new(index as f64 + 7., -(index as f64) - 11.))
                .collect(),
        }
    }

    #[test]
    fn checked_supplied_constructor_preserves_settings_without_admitting_cartesian_new() {
        for (n, h) in [(2, -2), (6, 1), (17, 4)] {
            for algorithm in [Algorithm::Ibp, Algorithm::PassarinoVeltman, Algorithm::Hybrid] {
                for convention in [Convention::IregiNative, Convention::OneLoop, Convention::MadLoop] {
                    let mut settings = ReductionSettings::new(n, 6);
                    settings.powers = vec![3; n];
                    settings.dimension_shift = h;
                    settings.algorithm = algorithm;
                    settings.convention = convention;
                    settings.tensor_weights = TensorWeights::Multinomial;
                    settings.invariants = InvariantInput::SuppliedScalarProducts;
                    let remaining = vec![2; n - 1];
                    let permutation = (0..n - 1).collect::<Vec<_>>();
                    let terminals = [terminal(n - 1, &remaining, &permutation)];
                    assert!(PowerBatchLayout::new(&settings, &terminals).is_err());
                    let before = serde_json::to_value(&settings).unwrap();
                    let layout = PowerBatchLayout::new_checked_supplied(&settings, &terminals).unwrap();
                    assert_eq!(serde_json::to_value(&settings).unwrap(), before);
                    let child = &layout.child_settings()[0];
                    assert_eq!(child.propagators, n - 1);
                    assert_eq!(child.powers, remaining);
                    assert_eq!(child.rank, settings.rank);
                    assert_eq!(child.dimension_shift, h);
                    assert_eq!(child.algorithm, algorithm);
                    assert_eq!(child.convention, convention);
                    assert_eq!(child.tensor_weights, settings.tensor_weights);
                    assert_eq!(child.invariants, InvariantInput::SuppliedScalarProducts);
                    assert_eq!(child.input_policy, InputPolicy::Preserve);
                    assert_eq!(layout.parent_layout.shifts, 0..4 * n);
                    assert_eq!(layout.parent_layout.scalar_products, Some(4 * n..4 * n + n * (n + 1) / 2));
                    assert_eq!(layout.parent_layout.masses.len(), n);
                    assert_eq!(layout.parent_parameters, layout.parent_layout.scale + 1);
                    let packed = layout.pack(&supplied_input(n, 1, 1), &[vec![0]], 1).unwrap();
                    assert_eq!(packed.groups[0].input.values.len(), layout.child_parameters);
                }
            }
        }
        let mut settings = ReductionSettings::new(3, 2);
        let terminals = [terminal(0, &[1, 1], &[0, 1])];
        assert!(PowerBatchLayout::new_checked_supplied(&settings, &terminals).is_err());
        assert!(PowerBatchLayout::new(&settings, &terminals).is_ok());
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        settings.input_policy = InputPolicy::legacy_onshell();
        assert!(PowerBatchLayout::new_checked_supplied(&settings, &terminals).is_err());
        settings.input_policy = InputPolicy::Preserve;
        settings.powers[1] = 0;
        assert!(PowerBatchLayout::new_checked_supplied(&settings, &terminals).is_err());
        settings.powers = vec![1];
        assert!(PowerBatchLayout::new_checked_supplied(&settings, &terminals).is_err());
        assert!(input_layout(usize::MAX, InvariantInput::SuppliedScalarProducts).is_err());
    }

    #[test]
    fn supplied_upper_triangle_follows_canonical_permutations_all_complex_components_and_lanes() {
        let mut settings = ReductionSettings::new(4, 6);
        settings.powers = vec![3; 4];
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        let terminals = [
            terminal(0, &[1, 3, 2], &[1, 2, 0]),
            terminal(2, &[3, 2, 1], &[0, 1, 2]),
            terminal(1, &[1, 1, 1], &[0, 1, 2]),
        ];
        let layout = PowerBatchLayout::new_checked_supplied(&settings, &terminals).unwrap();
        assert_eq!(layout.parent_parameters, 31);
        assert_eq!(layout.child_parameters, 22);
        let source = supplied_input(4, 3, 4);
        let original = source.values.clone();
        let demands = [vec![2, 1], vec![0], vec![1, 0]];
        let packed = layout.pack(&source, &demands, 5).unwrap();
        assert_eq!(source.values, original);
        assert_eq!(packed.total_rows, 5);
        assert_eq!(packed.groups[0].input.rows, 4);
        assert_eq!(packed.groups[1].input.rows, 1);
        let canonical_lines = [vec![2, 3, 1], vec![0, 1, 3], vec![0, 2, 3]];
        // Independently listed parent-G positions for each canonical triangle.
        // In the first map G(2,1) and G(3,1) reverse original index order.
        let canonical_products = [[7, 8, 5, 9, 6, 4], [0, 1, 3, 4, 6, 9], [0, 2, 3, 7, 8, 9]];
        for (group_index, group) in packed.groups.iter().enumerate() {
            assert_eq!(group.input.components, 4);
            for (child_row, origin) in group.origins.iter().enumerate() {
                let lines = &canonical_lines[origin.terminal_index];
                let parameters = lines.iter().flat_map(|&line| 4 * line..4 * line + 4)
                    .chain(canonical_products[origin.terminal_index].into_iter().map(|index| 16 + index))
                    .chain(lines.iter().map(|&line| 26 + line)).chain([30]).collect::<Vec<_>>();
                assert_eq!(parameters.len(), 22);
                assert_eq!(packed.require_location(origin.parent_row, origin.terminal_index).unwrap(),
                    PackedChildRow { group: group_index, child_row });
                for (argument, parameter) in parameters.into_iter().enumerate() {
                    assert_eq!(
                        &group.input.values[(child_row * 22 + argument) * 4..(child_row * 22 + argument + 1) * 4],
                        &source.values[(origin.parent_row * 31 + parameter) * 4..(origin.parent_row * 31 + parameter + 1) * 4],
                    );
                }
            }
        }
        assert!(packed.location(0, 0).is_none());
        assert!(packed.location(1, 1).is_none());
        assert!(packed.require_location(1, 2).is_err());
        let reordered_demands = [vec![1, 2], vec![0], vec![0, 1]];
        let reordered = layout.pack(&source, &reordered_demands, 5).unwrap();
        for (actual, expected) in reordered.groups.iter().zip(&packed.groups) {
            assert_eq!(actual.input.values, expected.input.values);
            assert_eq!(actual.origins, expected.origins);
        }
        let lane_order = [2, 0, 1];
        let reordered_source = KinematicsBatch {
            rows: 3,
            components: 4,
            values: lane_order.iter().flat_map(|&lane|
                source.values[lane * 31 * 4..(lane + 1) * 31 * 4].iter().cloned()).collect(),
        };
        let reordered_profiles = lane_order.iter().map(|&lane| demands[lane].clone()).collect::<Vec<_>>();
        let reordered_rows = layout.pack(&reordered_source, &reordered_profiles, 5).unwrap();
        for group in &reordered_rows.groups {
            for (child_row, origin) in group.origins.iter().enumerate() {
                for (argument, product) in canonical_products[origin.terminal_index].into_iter().enumerate() {
                    let original_lane = lane_order[origin.parent_row];
                    assert_eq!(
                        &group.input.values[(child_row * 22 + 12 + argument) * 4..(child_row * 22 + 13 + argument) * 4],
                        &source.values[(original_lane * 31 + 16 + product) * 4..(original_lane * 31 + 17 + product) * 4],
                    );
                }
            }
        }
        for malformed in [vec![vec![0, 0], vec![], vec![]], vec![vec![3], vec![], vec![]]] {
            assert!(layout.pack(&source, &malformed, 5).is_err());
        }
        assert!(layout.pack(&source, &demands, 4).is_err());
        let cartesian_width_only = input(4, 3, 4);
        assert!(layout.pack(&cartesian_width_only, &demands, 5).is_err());
        for (source, demands) in [
            (supplied_input(4, 0, 4), vec![]),
            (supplied_input(4, 3, 4), vec![vec![], vec![], vec![]]),
        ] {
            let empty = layout.pack(&source, &demands, 0).unwrap();
            assert_eq!(empty.total_rows, 0);
            assert!(empty.groups.iter().all(|group| group.input.values.is_empty() && group.origins.is_empty()));
        }
    }
}
