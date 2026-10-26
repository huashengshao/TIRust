//! Bounded metadata DAG for the existing constant-one routing recurrence.
//!
//! `ibp::ScalarCompiler::compile_routing_relation` already implements
//! `I(nu) = sum_i b_i I(nu-e_i)` when `sum_i b_i D_i = 1`. This adapter
//! changes only structural scheduling: equal power vectors share a node and
//! each branch stops at its first pinch. It neither derives a relation nor
//! implements polynomial arithmetic, a matrix solver, or a numerical zero test.
//!
//! The caller certifies inactive coefficients in *every* requested Taylor
//! component before construction. No primal-value pruning occurs here. The
//! numerator, integration variable, dimension shift and tensor basis do not
//! change along an edge; those belong to the surrounding reduction plan.
//!
//! Reuse checks: the recurrence above is the existing IREGI compiler path;
//! Symbolica `Atom` sums/products/aliases are used exactly as in `plan.rs` and
//! `pv.rs`; FeynKit's existing `IntegralFamily::partial_fraction` (extracted
//! locally as `AffineFamily`) owns general affine decomposition. This module
//! supplies only the narrower positive-power
//! scheduling required after an already certified constant-one relation.

use std::{cmp::Reverse, collections::HashMap};

use symbolica::atom::{Atom, AtomCore};

/// Independent, explicit caps. Exceeding any cap returns an error, never a
/// partially prepared decomposition. Counts include terminal nodes and edges
/// leading to already interned states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DagBudget {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_terminals: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Edge {
    /// Coefficient index in the original propagator order.
    pub line: usize,
    /// Always smaller than the containing node's index.
    pub child: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum NodeKind {
    Branch(Vec<Edge>),
    Terminal(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Node {
    /// Full original-line order, including the unique zero of a terminal.
    pub powers: Vec<u16>,
    pub kind: NodeKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Terminal {
    pub removed_line: usize,
    /// Original line order with `removed_line` deleted; all entries are positive.
    pub remaining_powers: Vec<u16>,
    /// Canonical position -> position in `remaining_powers`. Its stable key is
    /// `(Reverse(power), original_line_index)`. This is only a packing map;
    /// neither the integration variable nor the requested moments are shifted.
    pub canonical_to_remaining: Vec<usize>,
}

impl Terminal {
    pub fn canonical_powers(&self) -> Vec<u16> {
        self.canonical_to_remaining
            .iter()
            .map(|&position| self.remaining_powers[position])
            .collect()
    }

    pub fn canonical_original_lines(&self) -> Vec<usize> {
        self.canonical_to_remaining
            .iter()
            .map(|&position| position + usize::from(position >= self.removed_line))
            .collect()
    }

    fn from_pinched(powers: &[u16]) -> Result<Self, String> {
        let removed_line = powers
            .iter()
            .position(|&power| power == 0)
            .ok_or("affine terminal has no pinched line")?;
        if powers[removed_line + 1..].contains(&0) {
            return Err("affine terminal has more than one pinched line".into());
        }
        let remaining_powers: Vec<_> = powers
            .iter()
            .enumerate()
            .filter_map(|(line, &power)| (line != removed_line).then_some(power))
            .collect();
        let mut canonical_to_remaining: Vec<_> = (0..remaining_powers.len()).collect();
        canonical_to_remaining.sort_unstable_by_key(|&position| {
            (
                Reverse(remaining_powers[position]),
                position + usize::from(position >= removed_line),
            )
        });
        Ok(Self {
            removed_line,
            remaining_powers,
            canonical_to_remaining,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct AffinePowerDag {
    /// Deterministic topological order, children before parents.
    pub nodes: Vec<Node>,
    pub root: usize,
    pub terminals: Vec<Terminal>,
    pub edge_count: usize,
    original_lines: usize,
}

#[derive(Clone, Debug)]
pub(super) struct DagEmission {
    pub root: Atom,
    /// Child-before-parent aliases suitable for the existing evaluator builder.
    pub aliases: Vec<(Atom, Atom)>,
}

impl AffinePowerDag {
    /// `inactive_lines` are relation coefficients proved identically zero in
    /// the complete selected Taylor layout, not absent input propagators.
    /// Their positive powers remain in every state and every child integral.
    pub fn new(
        root_powers: Vec<u16>,
        inactive_lines: &[usize],
        budget: DagBudget,
    ) -> Result<Self, String> {
        let original_lines = root_powers.len();
        if original_lines < 2 || root_powers.contains(&0) {
            return Err("affine power root requires at least two strictly positive powers".into());
        }
        if budget.max_nodes == 0 || budget.max_edges == 0 || budget.max_terminals == 0 {
            return Err("affine power node, edge and terminal budgets must be positive".into());
        }
        let root_level = checked_total(&root_powers)?;
        let mut inactive = vec![false; original_lines];
        for &line in inactive_lines {
            let slot = inactive
                .get_mut(line)
                .ok_or("affine inactive line is outside the original topology")?;
            if *slot {
                return Err("affine inactive line is repeated".into());
            }
            *slot = true;
        }
        let active: Vec<_> = (0..original_lines)
            .filter(|&line| !inactive[line])
            .collect();
        if active.is_empty() {
            return Err(
                "constant-one affine relation cannot have all coefficients inactive".into(),
            );
        }

        // Iterative discovery avoids recursive host-stack growth for large
        // powers. Every edge lowers the checked total by exactly one.
        claim_count(0, budget.max_nodes, "node")?;
        let mut interned = HashMap::from([(root_powers.clone(), 0usize)]);
        let mut nodes = vec![Node {
            powers: root_powers,
            kind: NodeKind::Branch(Vec::new()),
        }];
        let mut levels = vec![root_level];
        let mut terminals = Vec::new();
        let mut edge_count = 0usize;
        let mut cursor = 0usize;
        while cursor < nodes.len() {
            if matches!(nodes[cursor].kind, NodeKind::Terminal(_)) {
                cursor += 1;
                continue;
            }
            let powers = nodes[cursor].powers.clone();
            let child_level = levels[cursor]
                .checked_sub(1)
                .ok_or("affine power level underflow")?;
            let mut edges = Vec::new();
            for &line in &active {
                edge_count = claim_count(edge_count, budget.max_edges, "edge")?;
                let mut child_powers = powers.clone();
                child_powers[line] = child_powers[line]
                    .checked_sub(1)
                    .ok_or("affine recurrence attempted to lower an already pinched line")?;
                let child = if let Some(&child) = interned.get(&child_powers) {
                    child
                } else {
                    claim_count(nodes.len(), budget.max_nodes, "node")?;
                    let child = nodes.len();
                    let kind = if child_powers[line] == 0 {
                        claim_count(terminals.len(), budget.max_terminals, "terminal")?;
                        let terminal = terminals.len();
                        terminals.push(Terminal::from_pinched(&child_powers)?);
                        NodeKind::Terminal(terminal)
                    } else {
                        NodeKind::Branch(Vec::new())
                    };
                    interned.insert(child_powers.clone(), child);
                    nodes.push(Node {
                        powers: child_powers,
                        kind,
                    });
                    levels.push(child_level);
                    child
                };
                edges.push(Edge { line, child });
            }
            nodes[cursor].kind = NodeKind::Branch(edges);
            cursor += 1;
        }

        // Discovery order is deterministic. Preserve it within equal levels,
        // then remap the node references to child-before-parent order.
        let mut order: Vec<_> = (0..nodes.len()).collect();
        order.sort_unstable_by_key(|&old| (levels[old], old));
        let mut new_index = vec![0; nodes.len()];
        for (new, &old) in order.iter().enumerate() {
            new_index[old] = new;
        }
        let root = new_index[0];
        let mut sorted = Vec::with_capacity(nodes.len());
        for old in order {
            let mut node = nodes[old].clone();
            if let NodeKind::Branch(edges) = &mut node.kind {
                for edge in edges {
                    edge.child = new_index[edge.child];
                }
            }
            sorted.push(node);
        }
        Ok(Self {
            nodes: sorted,
            root,
            terminals,
            edge_count,
            original_lines,
        })
    }

    /// Emit the existing recurrence using Symbolica operations only. A terminal
    /// marker denotes the *same requested tensor moment* of its pinched child;
    /// call this separately for every output slot, or provide full output
    /// marker arrays at the surrounding plan layer. No coefficient collection
    /// or hand-written Taylor multiplication is performed.
    ///
    /// The caller owns `plan_name` and must keep it unique when merging alias
    /// lists from separate emissions. Names are a single ASCII identifier and
    /// are placed under a reserved implementation namespace. A name collision
    /// with any coefficient or terminal expression is explicitly rejected.
    pub fn emit(
        &self,
        coefficients: &[Atom],
        terminal_markers: &[Atom],
        plan_name: &str,
    ) -> Result<DagEmission, String> {
        if coefficients.len() != self.original_lines
            || terminal_markers.len() != self.terminals.len()
            || self.root >= self.nodes.len()
        {
            return Err(
                "affine DAG emission dimensions do not match its relation and terminals".into(),
            );
        }
        let mut characters = plan_name.chars();
        if !characters
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            || !characters.all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err("affine DAG plan name must be one nonempty ASCII identifier".into());
        }
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut aliases = Vec::new();
        for (index, node) in self.nodes.iter().enumerate() {
            match &node.kind {
                NodeKind::Terminal(terminal) => {
                    values.push(
                        terminal_markers
                            .get(*terminal)
                            .ok_or("affine DAG terminal index is outside its marker list")?
                            .clone(),
                    );
                }
                NodeKind::Branch(edges) => {
                    if edges.is_empty() {
                        return Err("affine DAG contains an empty recurrence node".into());
                    }
                    let mut terms = Vec::with_capacity(edges.len());
                    for edge in edges {
                        if edge.child >= index || edge.line >= coefficients.len() {
                            return Err(
                                "affine DAG recurrence is not in child-before-parent order".into(),
                            );
                        }
                        terms.push(&coefficients[edge.line] * &values[edge.child]);
                    }
                    let alias = Atom::var(symbolica::symbol!(format!(
                        "iregi::affine_power::{plan_name}::node_{index}"
                    )));
                    if coefficients
                        .iter()
                        .chain(terminal_markers)
                        .any(|input| input.contains(alias.as_view()))
                    {
                        return Err(
                            "affine DAG reserved alias collides with an input expression".into(),
                        );
                    }
                    aliases.push((alias.clone(), Atom::add_many(&terms)));
                    values.push(alias);
                }
            }
        }
        Ok(DagEmission {
            root: values[self.root].clone(),
            aliases,
        })
    }
}

fn checked_total(powers: &[u16]) -> Result<usize, String> {
    powers.iter().try_fold(0usize, |sum, &power| {
        sum.checked_add(usize::from(power))
            .ok_or_else(|| "affine total power overflow".into())
    })
}

fn claim_count(current: usize, limit: usize, label: &str) -> Result<usize, String> {
    let next = current
        .checked_add(1)
        .ok_or_else(|| format!("affine {label} count overflow"))?;
    if next > limit {
        return Err(format!("affine {label} budget exceeded ({next}>{limit})"));
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    //! Source-only proposal tests: deliberately UNEXECUTED in this checkpoint.
    use super::*;
    use std::collections::HashSet;
    use symbolica::atom::AtomView;

    fn budget() -> DagBudget {
        DagBudget {
            max_nodes: 10_000,
            max_edges: 100_000,
            max_terminals: 10_000,
        }
    }

    fn variable(name: &str) -> Atom {
        Atom::var(symbolica::symbol!(format!(
            "affine_power_dag_tests::{name}"
        )))
    }

    fn expand(emission: &DagEmission) -> Atom {
        let mut result = emission.root.clone();
        for pair in emission.aliases.iter().rev() {
            result = result.replace_multiple(std::iter::once(symbolica::id::Replacement::new(pair.0.clone(),pair.1.clone())));
        }
        result
    }

    #[test]
    fn first_dot_six_line_has_both_terminal_power_families() {
        let dag = AffinePowerDag::new(vec![2, 1, 1, 1, 1, 1], &[], budget()).unwrap();
        assert_eq!(
            (dag.nodes.len(), dag.edge_count, dag.terminals.len()),
            (13, 12, 11)
        );
        assert_eq!(
            dag.terminals
                .iter()
                .filter(|t| t.canonical_powers() == [1; 5])
                .count(),
            6
        );
        assert_eq!(
            dag.terminals
                .iter()
                .filter(|t| t.canonical_powers() == [2, 1, 1, 1, 1])
                .count(),
            5
        );
        for terminal in &dag.terminals {
            let lines = terminal.canonical_original_lines();
            assert!(!lines.contains(&terminal.removed_line));
            assert_eq!(lines.len(), 5);
        }
    }

    #[test]
    fn triple_and_mixed_powers_share_states_and_have_topological_edges() {
        let dag = AffinePowerDag::new(vec![3, 2, 1], &[], budget()).unwrap();
        assert_eq!(
            (dag.nodes.len(), dag.edge_count, dag.terminals.len()),
            (17, 18, 11)
        );
        assert_eq!(
            dag.nodes
                .iter()
                .map(|n| n.powers.clone())
                .collect::<HashSet<_>>()
                .len(),
            dag.nodes.len()
        );
        let shared = dag
            .nodes
            .iter()
            .position(|n| n.powers == [2, 1, 1])
            .unwrap();
        let mut incoming = 0;
        for (parent, node) in dag.nodes.iter().enumerate() {
            if let NodeKind::Branch(edges) = &node.kind {
                for edge in edges {
                    assert!(edge.child < parent);
                    let mut expected = node.powers.clone();
                    expected[edge.line] -= 1;
                    assert_eq!(dag.nodes[edge.child].powers, expected);
                    incoming += usize::from(edge.child == shared);
                }
            }
        }
        assert_eq!(incoming, 2);
        assert_eq!(dag.nodes[dag.root].powers, [3, 2, 1]);
    }

    #[test]
    fn arbitrary_seventeen_line_metadata_does_not_copy_reference_limits() {
        let dag = AffinePowerDag::new(vec![1; 17], &[], budget()).unwrap();
        assert_eq!(
            (dag.nodes.len(), dag.edge_count, dag.terminals.len()),
            (18, 17, 17)
        );
        assert!(dag.terminals.iter().all(|t| t.remaining_powers == [1; 16]));
        assert_eq!(
            dag.terminals
                .iter()
                .map(|t| t.removed_line)
                .collect::<HashSet<_>>()
                .len(),
            17
        );
    }

    #[test]
    fn certified_inactive_coefficients_keep_their_denominators_and_powers() {
        let dag = AffinePowerDag::new(vec![2, 4, 1, 3], &[1, 3], budget()).unwrap();
        assert_eq!(
            (dag.nodes.len(), dag.edge_count, dag.terminals.len()),
            (5, 4, 3)
        );
        for node in &dag.nodes {
            assert_eq!((node.powers[1], node.powers[3]), (4, 3));
            if let NodeKind::Branch(edges) = &node.kind {
                assert_eq!(edges.iter().map(|e| e.line).collect::<Vec<_>>(), [0, 2]);
            }
        }
        for terminal in &dag.terminals {
            assert!([0, 2].contains(&terminal.removed_line));
            assert_eq!(terminal.canonical_powers()[..2], [4, 3]);
        }
    }

    #[test]
    fn invalid_roots_inactive_sets_and_each_resource_budget_fail_explicitly() {
        for root in [vec![], vec![1], vec![1, 0], vec![0, 1]] {
            assert!(AffinePowerDag::new(root, &[], budget()).is_err());
        }
        for inactive in [vec![2], vec![0, 0], vec![0, 1]] {
            assert!(AffinePowerDag::new(vec![1, 1], &inactive, budget()).is_err());
        }
        for limited in [
            DagBudget {
                max_nodes: 3,
                ..budget()
            },
            DagBudget {
                max_edges: 2,
                ..budget()
            },
            DagBudget {
                max_terminals: 2,
                ..budget()
            },
            DagBudget {
                max_nodes: 0,
                ..budget()
            },
            DagBudget {
                max_edges: 0,
                ..budget()
            },
            DagBudget {
                max_terminals: 0,
                ..budget()
            },
        ] {
            assert!(AffinePowerDag::new(vec![1; 3], &[], limited).is_err());
        }
        assert!(
            claim_count(usize::MAX, usize::MAX, "test")
                .unwrap_err()
                .contains("overflow")
        );
        assert_eq!(
            checked_total(&[u16::MAX; 3]).unwrap(),
            3 * usize::from(u16::MAX)
        );
        assert!(
            AffinePowerDag::new(
                vec![u16::MAX, 1],
                &[],
                DagBudget {
                    max_nodes: 2,
                    ..budget()
                }
            )
            .is_err()
        );
    }

    fn rational_fixture() -> (Vec<Atom>, Vec<Atom>, symbolica::atom::Symbol) {
        // D0 and D1 have equal shifts and independent squared masses. Their
        // difference provides a symbolic constant-one relation. The remaining
        // four lines are present but have structurally zero relation weights.
        let qd2 = variable("qd2");
        let q0 = variable("q0");
        let masses: Vec<_> = (0..6).map(|i| variable(&format!("m2_{i}"))).collect();
        let denominators: Vec<_> = masses
            .iter()
            .enumerate()
            .map(|(i, mass)| {
                if i < 2 {
                    &qd2 - mass
                } else {
                    &qd2 + Atom::num(i as i64) * &q0 + Atom::num((i * i) as i64) - mass
                }
            })
            .collect();
        let delta = &masses[1] - &masses[0];
        let mut coefficients = vec![Atom::new(); 6];
        coefficients[0] = Atom::num(1) / &delta;
        coefficients[1] = -Atom::num(1) / &delta;
        let AtomView::Var(mass) = masses[0].as_view() else {
            unreachable!()
        };
        (denominators, coefficients, mass.get_symbol())
    }

    fn integrand(denominators: &[Atom], powers: &[u16]) -> Atom {
        denominators
            .iter()
            .zip(powers)
            .map(|(d, &p)| d.pow(-i64::from(p)))
            .product()
    }

    fn emitted_integrand(
        dag: &AffinePowerDag,
        denominators: &[Atom],
        coefficients: &[Atom],
        name: &str,
    ) -> Atom {
        let terminals: Vec<_> = dag
            .terminals
            .iter()
            .map(|terminal| {
                denominators
                    .iter()
                    .enumerate()
                    .filter(|(line, _)| *line != terminal.removed_line)
                    .zip(&terminal.remaining_powers)
                    .map(|((_, denominator), &power)| denominator.pow(-i64::from(power)))
                    .product::<Atom>()
            })
            .collect();
        expand(&dag.emit(coefficients, &terminals, name).unwrap())
    }

    #[test]
    fn exact_symbolica_integrand_identity_keeps_the_full_tensor_numerator() {
        let (denominators, coefficients, _) = rational_fixture();
        let constant: Atom = coefficients
            .iter()
            .zip(&denominators)
            .map(|(b, d)| b * d)
            .sum();
        assert!((constant - Atom::num(1)).together().is_zero());
        let root = vec![3, 2, 1, 1, 1, 1];
        let dag = AffinePowerDag::new(root.clone(), &[2, 3, 4, 5], budget()).unwrap();
        let numerator = variable("q0").pow(3) * variable("q1");
        let reduced = emitted_integrand(&dag, &denominators, &coefficients, "integrand");
        assert!(
            (&numerator * (reduced - integrand(&denominators, &root)))
                .together()
                .is_zero()
        );
    }

    #[test]
    fn one_dot_equals_independent_mass_derivative_including_coefficient_derivatives() {
        let (denominators, coefficients, mass) = rational_fixture();
        assert!(!coefficients[0].derivative(mass).together().is_zero());
        let unit = vec![1; 6];
        let dotted = vec![2, 1, 1, 1, 1, 1];
        let unit_dag = AffinePowerDag::new(unit.clone(), &[2, 3, 4, 5], budget()).unwrap();
        let dot_dag = AffinePowerDag::new(dotted.clone(), &[2, 3, 4, 5], budget()).unwrap();
        let reduced_unit = emitted_integrand(&unit_dag, &denominators, &coefficients, "mass_unit");
        let reduced_dot = emitted_integrand(&dot_dag, &denominators, &coefficients, "mass_dot");
        let independent = integrand(&denominators, &unit).derivative(mass);
        assert!(
            (&independent - integrand(&denominators, &dotted))
                .together()
                .is_zero()
        );
        assert!((&reduced_dot - &independent).together().is_zero());
        assert!(
            (reduced_unit.derivative(mass) - independent)
                .together()
                .is_zero()
        );
    }

    #[test]
    fn emission_checks_layout_namespace_and_alias_capture_without_primal_pruning() {
        let dag = AffinePowerDag::new(vec![1, 1], &[], budget()).unwrap();
        let coefficients = vec![variable("b0"), variable("b1")];
        let terminals = vec![variable("child0"), variable("child1")];
        let emitted = dag.emit(&coefficients, &terminals, "ordinary").unwrap();
        assert_eq!(emitted.aliases.len(), 1);
        let result = expand(&emitted);
        assert!(result.contains(coefficients[0].as_view()));
        assert!(result.contains(coefficients[1].as_view()));
        assert!(
            dag.emit(&coefficients[..1], &terminals, "bad_width")
                .is_err()
        );
        assert!(
            dag.emit(&coefficients, &terminals[..1], "bad_terminals")
                .is_err()
        );
        for name in ["", "1plan", "plan::nested", "bad-name"] {
            assert!(dag.emit(&coefficients, &terminals, name).is_err());
        }
        let captured = vec![emitted.root.clone(), variable("safe")];
        assert!(
            dag.emit(&captured, &terminals, "ordinary")
                .unwrap_err()
                .contains("collides")
        );
    }
}
