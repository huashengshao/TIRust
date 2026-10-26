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
use crate::{InputPolicy, InvariantInput, KinematicsBatch, ReductionError, ReductionSettings};

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
}

/// Structural child groups. No point-count, rank or h restriction is imposed
/// by this transport helper. The initial interface requires Preserve and
/// FromShifts so a denominator permutation never changes an input policy.
#[derive(Clone, Debug)]
pub(super) struct PowerBatchLayout {
    propagators: usize,
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
        let n = settings.propagators;
        if n < 2
            || settings.powers.len() != n
            || settings.powers.contains(&0)
            || settings.invariants != InvariantInput::FromShifts
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(invalid(
                "affine-power child packing requires N>=2, N positive powers, FromShifts and Preserve",
            ));
        }
        let parent_parameters = parameters(n)?;
        let child_parameters = parameters(n - 1)?;
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
            });
        }
        Ok(Self {
            propagators: n,
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
                for &line in &terminal.original_lines {
                    copy(4 * self.propagators + line);
                }
                copy(5 * self.propagators);
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
}
