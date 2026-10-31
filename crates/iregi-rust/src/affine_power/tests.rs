//! Source proposal tests. UNEXECUTED: these are not numerical qualification.

use super::*;
use crate::{Convention, TensorWeights};
use std::time::Duration;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Float, FloatLike},
};

fn symbol(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!(
        "affine_power_wrapper_tests::{name}"
    )))
}

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn small_plan() -> AffinePowerReductionPlan {
    AffinePowerReductionPlan::new(ReductionSettings::new(3, 1)).unwrap()
}

fn prime_observables(workspace: &mut AffinePowerWorkspace<f64>) {
    workspace.stats = AffinePowerExecutionStats {
        input_rows: 71,
        profiles: 72,
        dag_nodes: 73,
        dag_edges: 74,
        packed_child_rows: 75,
        child_batch_calls: 76,
        parent_batch_calls: 77,
        parent_rows: 78,
        quadrature_work_rows: 79,
        certificate: CertificateExecutionStats {
            input_rows: 80,
            evaluated_rows: 81,
            passes: 82,
            reduction_batches: 83,
        },
        child_execution: ExecutionStats {
            master_rows: 84,
            ..Default::default()
        },
        parent_execution: ExecutionStats {
            reduction_rows: 85,
            ..Default::default()
        },
        child_methods: vec![AffinePowerChildMethod {
            propagators: 2,
            powers: vec![1, 1],
            requested: Algorithm::Ibp,
            actual: Some(Algorithm::PassarinoVeltman),
            rows: 86,
            notes: vec!["stale earlier method".into()],
        }],
    };
    workspace.diagnostics = vec![vec!["stale earlier lane".into()]];
    workspace.profile = ExecutionProfile {
        arguments: Duration::from_secs(90),
        scalar_masters: Duration::from_secs(91),
        reduction: Duration::from_secs(92),
        total: Duration::from_secs(93),
    };
}

fn assert_preflight_only(workspace: &AffinePowerWorkspace<f64>, rows: usize) {
    let stats = workspace.last_execution();
    assert_eq!(stats.input_rows, rows);
    assert_eq!(stats.certificate, CertificateExecutionStats::default());
    assert_eq!(stats.profiles, 0);
    assert_eq!(stats.dag_nodes, 0);
    assert_eq!(stats.dag_edges, 0);
    assert_eq!(stats.packed_child_rows, 0);
    assert_eq!(stats.child_batch_calls, 0);
    assert_eq!(stats.parent_batch_calls, 0);
    assert_eq!(stats.parent_rows, 0);
    assert_eq!(stats.quadrature_work_rows, 0);
    assert_eq!(stats.child_execution, ExecutionStats::default());
    assert_eq!(stats.parent_execution, ExecutionStats::default());
    assert!(stats.child_methods.is_empty());
    assert!(workspace.last_diagnostics().iter().all(Vec::is_empty));
    let profile = workspace.last_profile().unwrap();
    assert_eq!(profile.arguments, Duration::ZERO);
    assert_eq!(profile.scalar_masters, Duration::ZERO);
    assert_eq!(profile.reduction, Duration::ZERO);
    // An error may retain its own elapsed preflight time, never the old call.
    assert_ne!(profile.total, Duration::from_secs(93));
    assert!(workspace.profiles.is_empty());
    assert!(workspace.children.is_empty());
}

#[test]
fn dynamic_outer_constructor_preserves_request_before_any_inner_point_guard() {
    for (h, algorithm, convention, weights) in [
        (
            -1,
            Algorithm::Ibp,
            Convention::IregiNative,
            TensorWeights::Components,
        ),
        (
            1,
            Algorithm::PassarinoVeltman,
            Convention::OneLoop,
            TensorWeights::Multinomial,
        ),
        (
            2,
            Algorithm::Hybrid,
            Convention::MadLoop,
            TensorWeights::Components,
        ),
    ] {
        let mut settings = ReductionSettings::new(17, 6);
        settings.powers[0] = 3;
        settings.powers[8] = 2;
        settings.dimension_shift = h;
        settings.algorithm = algorithm;
        settings.convention = convention;
        settings.tensor_weights = weights;
        let budget = AffinePowerBudget {
            max_cofactors: 17,
            max_parent_parameters: 86,
            max_packed_child_rows: 3,
            ..Default::default()
        };
        let plan = AffinePowerReductionPlan::with_budget(settings.clone(), budget).unwrap();
        assert_eq!(plan.settings().propagators, 17);
        assert_eq!(plan.settings().powers, settings.powers);
        assert_eq!(plan.settings().rank, 6);
        assert_eq!(plan.settings().dimension_shift, h);
        assert_eq!(plan.settings().algorithm, algorithm);
        assert_eq!(plan.settings().convention, convention);
        assert_eq!(plan.settings().tensor_weights, weights);
        assert_eq!(plan.settings().input_policy, InputPolicy::Preserve);
        assert_eq!(plan.settings().invariants, InvariantInput::FromShifts);
        assert_eq!(plan.basis(), tensor_basis(6));
        assert_eq!(plan.parameter_count(), 86);
        assert_eq!(plan.input_layout().shifts, 0..68);
        assert_eq!(plan.input_layout().masses, 68..85);
        assert_eq!(plan.input_layout().scale, 85);
        assert_eq!(plan.budget.max_packed_child_rows, 3);
        assert!(plan.guards.lock().unwrap().is_empty());
        assert!(plan.profiles.lock().unwrap().is_empty());
        assert!(plan.children.lock().unwrap().is_empty());
    }
}

#[test]
fn malformed_requests_rank_guard_and_each_zero_or_geometry_budget_fail_explicitly() {
    let good = ReductionSettings::new(3, 1);
    let mut bad = good.clone();
    bad.propagators = 1;
    assert!(matches!(
        AffinePowerReductionPlan::new(bad),
        Err(ReductionError::InvalidInput(_))
    ));
    let mut bad = good.clone();
    bad.powers.pop();
    assert!(AffinePowerReductionPlan::new(bad).is_err());
    let mut bad = good.clone();
    bad.powers[1] = 0;
    assert!(AffinePowerReductionPlan::new(bad).is_err());
    let mut bad = good.clone();
    bad.invariants = InvariantInput::SuppliedScalarProducts;
    assert!(AffinePowerReductionPlan::new(bad).is_err());
    let mut bad = good.clone();
    bad.input_policy = InputPolicy::LegacyOnShell {
        threshold: "0".into(),
    };
    assert!(AffinePowerReductionPlan::new(bad).is_err());
    let mut bad = good.clone();
    bad.rank = 7;
    assert!(matches!(
        AffinePowerReductionPlan::new(bad),
        Err(ReductionError::Unsupported(_))
    ));
    for budget in [
        AffinePowerBudget {
            max_charts: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_constraints: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_cofactors: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_dag_nodes: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_dag_edges: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_terminals: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_parent_parameters: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_packed_child_rows: 0,
            ..Default::default()
        },
        AffinePowerBudget {
            max_parent_parameters: 15,
            ..Default::default()
        },
        AffinePowerBudget {
            max_cofactors: 2,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            AffinePowerReductionPlan::with_budget(good.clone(), budget),
            Err(ReductionError::InvalidInput(_))
        ));
    }
    assert!(parameters(usize::MAX).is_err());
}

#[test]
fn raw_preflight_and_empty_calls_reset_all_observables_without_child_or_master_work() {
    let plan = small_plan();
    let mut workspace = AffinePowerWorkspace::new(&plan, 0f64).unwrap();
    workspace.set_profiling(true);
    for malformed in [
        KinematicsBatch {
            rows: 3,
            components: 2,
            values: vec![],
        },
        KinematicsBatch {
            rows: 1,
            components: 1,
            values: vec![c(1.)],
        },
        KinematicsBatch {
            rows: usize::MAX,
            components: 1,
            values: vec![],
        },
    ] {
        prime_observables(&mut workspace);
        let rows = malformed.rows;
        assert!(workspace.reduce_coupled_batch(&malformed).is_err());
        assert_preflight_only(&workspace, rows);
    }
    prime_observables(&mut workspace);
    let empty = workspace
        .reduce_coupled_batch(&KinematicsBatch {
            rows: 0,
            components: 1,
            values: vec![],
        })
        .unwrap();
    assert_eq!(empty.result().rows(), 0);
    assert_eq!(empty.result().basis, tensor_basis(1));
    assert_eq!(empty.result().components, 1);
    assert_eq!(empty.result().metadata(), &workspace.metadata);
    assert!(empty.result().values.is_empty());
    assert!(empty.result().statuses.is_empty());
    assert!(empty.quadrature_estimates(0).is_err());
    assert_preflight_only(&workspace, 0);
}

#[test]
fn typed_length_workspace_and_ordered_shape_errors_also_clear_previous_call_metadata() {
    let plan = small_plan();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
    let mut workspace =
        AffinePowerWorkspace::new_dual(&plan, 0f64, HyperDual::<u8>::new(shape.clone())).unwrap();
    workspace.set_profiling(true);
    let empty: Vec<HyperDual<Complex<f64>>> = vec![];
    prime_observables(&mut workspace);
    assert!(workspace.reduce_typed_dual_batch(&empty, 1).is_err());
    assert_preflight_only(&workspace, 1);
    let reordered = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
    let input = vec![HyperDual::from_values(reordered, vec![c(1.), c(2.), c(3.), c(4.)]); 16];
    prime_observables(&mut workspace);
    assert!(workspace.reduce_typed_dual_batch(&input, 1).is_err());
    assert_preflight_only(&workspace, 1);
    let mut ordinary = AffinePowerWorkspace::new(&plan, 0f64).unwrap();
    ordinary.set_profiling(true);
    prime_observables(&mut ordinary);
    let input = vec![HyperDual::from_values(shape, vec![c(1.), c(2.), c(3.), c(4.)]); 16];
    assert!(ordinary.reduce_typed_dual_batch(&input, 1).is_err());
    assert_preflight_only(&ordinary, 1);
    prime_observables(&mut workspace);
    let result = workspace.reduce_typed_dual_batch(&empty, 0).unwrap();
    assert_eq!(result.rows(), 0);
    assert_eq!(result.components, 4);
    assert_preflight_only(&workspace, 0);
}

fn coupled_fixture() -> AffinePowerResult<Float> {
    // Transport-integrity fixture, not a physical integral or zero answer.
    // Distinct nonzero values cover every Laurent and mixed-component slot.
    let result = BatchResult {
        basis: vec![Monomial([0; 4]), Monomial([1, 0, 0, 0])],
        components: 4,
        metadata: ResultMetadata::new(
            Convention::IregiNative,
            TensorWeights::Components,
            Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
        ),
        values: (0..32)
            .map(|i| Complex::new(Float::with_val(192, i + 1), Float::with_val(160, -(i + 3))))
            .collect(),
        statuses: vec![LaneStatus::Success],
        fallback_used: vec![false],
        analytic_used: vec![true],
        primal_envelopes: vec![None],
        primal_envelope_errors: vec![None],
        estimated_errors: Vec::new(), best_effort: Vec::new(),
    };
    let errors: Vec<_> = (0..32)
        .map(|i| Complex::new(Float::with_val(256, i + 2), Float::with_val(256, 0)))
        .collect();
    let mut context = QuadratureErrorContext::missing(1);
    context
        .install_dense(
            &result,
            0,
            256,
            53,
            &errors,
            QuadratureErrorOrigin::PropagatedEuclidean,
        )
        .unwrap();
    AffinePowerResult {
        owned: CoupledQuadratureBatch::new(result, context).unwrap(),
        working: 256,
        accuracy: 53,
    }
}

#[test]
fn public_coupled_result_binds_all_slots_precision_and_ordered_interpretation() {
    let original = coupled_fixture();
    assert_eq!(original.result().lane(0).unwrap().len(), 32);
    assert_eq!(original.quadrature_estimates(0).unwrap().len(), 32);
    for value in original.result().lane(0).unwrap() {
        assert_eq!(value.re.get_precision(), 192);
        assert_eq!(value.im.get_precision(), 160);
    }
    for index in 0..32 {
        let public = coupled_fixture();
        let (mut result, context) = public.owned.into_parts();
        result.values[index].re = Float::with_val(192, -7);
        let mutated = AffinePowerResult {
            owned: CoupledQuadratureBatch::new(result, context).unwrap(),
            working: 256,
            accuracy: 53,
        };
        assert!(
            mutated.quadrature_estimates(0).is_err(),
            "stale complex slot {index}"
        );
    }
    for mutation in 0..6 {
        let public = coupled_fixture();
        let (mut result, context) = public.owned.into_parts();
        match mutation {
            0 => result.values[31].re = result.values[31].re.at_precision(64),
            1 => {
                result.metadata = ResultMetadata::new(
                    Convention::OneLoop,
                    TensorWeights::Components,
                    Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
                )
            }
            2 => {
                result.metadata = ResultMetadata::new(
                    Convention::IregiNative,
                    TensorWeights::Multinomial,
                    Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
                )
            }
            3 => {
                result.metadata = ResultMetadata::new(
                    Convention::IregiNative,
                    TensorWeights::Components,
                    Some(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]),
                )
            }
            4 => result.basis[1] = Monomial([0, 1, 0, 0]),
            _ => result.statuses[0] = LaneStatus::Unstable("injected failed lane".into()),
        }
        let mutated = AffinePowerResult {
            owned: CoupledQuadratureBatch::new(result, context).unwrap(),
            working: 256,
            accuracy: 53,
        };
        assert!(
            mutated.quadrature_estimates(0).is_err(),
            "stale interpretation {mutation}"
        );
    }
    let public = coupled_fixture();
    let plain = public.into_result();
    assert_eq!(plain.lane(0).unwrap().len(), 32);
    // The consumed public result exposes no sidecar: only the internal owner
    // offers into_parts, used above solely for deliberate corruption tests.
}

fn geometry(n: usize) -> Vec<Atom> {
    (0..5 * n + 1)
        .map(|index| symbol(&format!("geometry_{n}_{index}")))
        .collect()
}

fn small_chart() -> crate::pv::AffineChartKey {
    crate::pv::AffineChartKey {
        coordinate_rows: vec![1],
        relative_columns: vec![0, 1],
    }
}

fn prepare_small(settings: &ReductionSettings, shape: &Shape) -> (ParentRecipe, AffinePowerDag) {
    let dag = AffinePowerDag::new(
        settings.powers.clone(),
        &[],
        AffinePowerBudget::default().dag(),
    )
    .unwrap();
    let recipe = ParentRecipe::prepare(
        settings,
        &small_chart(),
        &[],
        &dag,
        &geometry(3),
        shape,
        AffinePowerBudget::default().chart(),
        100_000,
    )
    .unwrap();
    (recipe, dag)
}

fn expand_aliases(mut expression: Atom, aliases: &[(Atom, Atom)]) -> Atom {
    for pair in aliases.iter().rev() {
        expression = expression.replace_multiple(std::iter::once(symbolica::id::Replacement::new(pair.0.clone(),pair.1.clone())));
    }
    expression
}

#[test]
fn actual_two_by_two_parent_matches_independent_fixed_chart_in_every_native_slot() {
    // D0=q²-1, D1=(q+ex)²-2, D2=(q+2ex)²-3 satisfy
    // -D0/2 + D1 - D2/2 = 1. q is unchanged for every tensor component.
    let fixed: [i64; 16] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 2, 3, 3];
    for (convention, weights) in [
        (Convention::IregiNative, TensorWeights::Components),
        (Convention::OneLoop, TensorWeights::Multinomial),
        (Convention::MadLoop, TensorWeights::Components),
    ] {
        for shape in [
            None,
            Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
        ] {
            let mut settings = ReductionSettings::new(3, 1);
            settings.dimension_shift = 1;
            settings.convention = convention;
            settings.tensor_weights = weights;
            let (recipe, dag) = prepare_small(&settings, &shape);
            assert_eq!(recipe.plan.settings.rank, 1);
            assert_eq!(recipe.plan.settings.dimension_shift, 1);
            assert_eq!(recipe.plan.settings.convention, convention);
            assert_eq!(recipe.plan.settings.tensor_weights, weights);
            assert_eq!(
                recipe.plan.analytic_jet_order,
                if shape.is_some() { 2 } else { 0 }
            );
            assert_eq!(recipe.geometry_parameters, 16);
            assert_eq!(dag.terminals.len(), 3);
            assert_eq!(recipe.plan.moments.len(), 5);
            let replacements: Vec<_> = recipe.plan.parameters[..16]
                .iter()
                .cloned()
                .zip(fixed.map(Atom::num))
                .collect();
            let by_removed: BTreeMap<_, _> = dag
                .terminals
                .iter()
                .enumerate()
                .map(|(i, t)| (t.removed_line, i))
                .collect();
            for (moment, slots) in recipe.plan.moments.iter().enumerate() {
                for (pole, actual) in slots.iter().enumerate() {
                    let output = moment * 4 + pole;
                    let child = |removed| {
                        recipe.plan.external_primal_seeds[by_removed[&removed] * 20 + output]
                            .0
                            .clone()
                    };
                    let expected = -child(0) / Atom::num(2) + child(1) - child(2) / Atom::num(2);
                    let actual = expand_aliases(actual.clone(), &recipe.plan.aliases)
                        .replace_multiple(replacements.iter().map(|(a,b)|symbolica::id::Replacement::new(a.clone(),b.clone())));
                    assert!(
                        (actual - expected).together().is_zero(),
                        "moment {moment} native slot {pole}"
                    );
                }
            }
            assert!(recipe.plan.leaves.is_empty());
            assert!(recipe.plan.master_arguments.is_empty());
        }
    }
}

#[test]
fn every_dotted_terminal_seed_uses_its_own_same_h_full_rank_mass_dimension() {
    for powers in [vec![2, 1, 1], vec![3, 1, 1], vec![2, 2, 1]] {
        for h in [-1, 0, 1] {
            let mut settings = ReductionSettings::new(3, 1);
            settings.powers = powers.clone();
            settings.dimension_shift = h;
            let (recipe, dag) = prepare_small(&settings, &None);
            assert_eq!(recipe.plan.settings.powers, powers);
            assert_eq!(recipe.plan.settings.dimension_shift, h);
            assert_eq!(recipe.plan.basis, tensor_basis(1));
            let imports = dag.terminals.len() * 20;
            assert_eq!(recipe.plan.external_primal_seeds.len(), imports);
            assert_eq!(recipe.plan.parameter_count(), 16 + 2 * imports);
            for (terminal_index, terminal) in dag.terminals.iter().enumerate() {
                let total: u16 = terminal.remaining_powers.iter().sum();
                // Independently tabulated dimensions D-2*sum(nu) for the
                // possible two-denominator child families in D=2,4,6.
                let scalar_dimension = match (h, total) {
                    (-1, 2) => -2,
                    (-1, 3) => -4,
                    (-1, 4) => -6,
                    (0, 2) => 0,
                    (0, 3) => -2,
                    (0, 4) => -4,
                    (1, 2) => 2,
                    (1, 3) => 0,
                    (1, 4) => -2,
                    _ => panic!("unexpected child family"),
                };
                for (moment, _) in recipe.plan.basis.iter().enumerate() {
                    let expected = scalar_dimension + i32::from(moment != 0);
                    for pole in 0..4 {
                        let index = terminal_index * 20 + moment * 4 + pole;
                        let (value, amplitude, dimension) =
                            &recipe.plan.external_primal_seeds[index];
                        assert_eq!(value, &recipe.plan.parameters[16 + index]);
                        assert_eq!(amplitude, &recipe.plan.parameters[16 + imports + index]);
                        assert_eq!(
                            *dimension, expected,
                            "terminal {terminal_index}, moment {moment}, slot {pole}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn parent_constructor_retains_d6_rank6_uv_and_never_reconverts_imported_conventions() {
    for h in [-1, 0, 1] {
        for convention in [
            Convention::IregiNative,
            Convention::OneLoop,
            Convention::MadLoop,
        ] {
            let mut settings = ReductionSettings::new(6, 6);
            settings.dimension_shift = h;
            settings.convention = convention;
            settings.tensor_weights = TensorWeights::Multinomial;
            let basis = tensor_basis(6);
            let moments: Vec<[Atom; 4]> = (0..basis.len())
                .map(|moment| {
                    std::array::from_fn(|pole| symbol(&format!("import_{moment}_{pole}")))
                })
                .collect();
            let mut parameters = geometry(6);
            let values: Vec<_> = moments.iter().flatten().cloned().collect();
            let amplitudes: Vec<_> = (0..values.len())
                .map(|i| symbol(&format!("amplitude_{i}")))
                .collect();
            parameters.extend(values.iter().cloned());
            parameters.extend(amplitudes.iter().cloned());
            let seeds: Vec<_> = values
                .into_iter()
                .zip(amplitudes)
                .enumerate()
                .map(|(index, (value, amplitude))| {
                    (
                        value,
                        amplitude,
                        basis[index / 4].rank() as i32 + 4 + 2 * i32::from(h) - 10,
                    )
                })
                .collect();
            let shifts: Vec<[Atom; 4]> = (0..6)
                .map(|line| std::array::from_fn(|mu| parameters[line * 4 + mu].clone()))
                .collect();
            let distances = shifts
                .iter()
                .map(|p| {
                    shifts
                        .iter()
                        .map(|q| crate::plan::squared_difference(p, q))
                        .collect()
                })
                .collect();
            let plan = ReductionPlan::new_affine_power_parent(
                settings.clone(),
                0,
                parameters,
                seeds,
                InputLayout {
                    shifts: 0..24,
                    scalar_products: None,
                    masses: 24..30,
                    scale: 30,
                },
                distances,
                moments.clone(),
                vec![],
                vec!["uninterpreted imported-output constructor test".into()],
            )
            .unwrap();
            assert_eq!(plan.settings.convention, convention);
            assert_eq!(plan.settings.tensor_weights, TensorWeights::Multinomial);
            assert_eq!(plan.settings.dimension_shift, h);
            assert_eq!(plan.algorithm_used(), Algorithm::Hybrid);
            for (index, (actual, imported)) in plan.moments.iter().zip(&moments).enumerate() {
                for pole in 1..4 {
                    assert_eq!(
                        actual[pole], imported[pole],
                        "second normalization/weight conversion at {index}:{pole}"
                    );
                }
                if h == 1 && basis[index].rank() == 6 {
                    assert_eq!(actual[0], imported[0]);
                    assert!(
                        !actual[0].is_zero(),
                        "D6 rank6 power counting cannot discard an imported UV residue"
                    );
                } else {
                    assert!(actual[0].is_zero());
                }
            }
        }
    }
}

#[test]
fn analytic_order_and_import_parameter_budget_precede_any_chart_solve() {
    assert_eq!(analytic_order(&None).unwrap(), 0);
    assert_eq!(
        analytic_order(&Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]])).unwrap(),
        2
    );
    assert!(analytic_order(&Some(vec![vec![usize::MAX, 1]])).is_err());
    assert!(analytic_order(&Some(vec![vec![usize::from(u16::MAX) + 1]])).is_err());
    let settings = ReductionSettings::new(3, 1);
    let dag = AffinePowerDag::new(vec![1; 3], &[], AffinePowerBudget::default().dag()).unwrap();
    let deliberately_bad_chart = crate::pv::AffineChartKey {
        coordinate_rows: vec![4],
        relative_columns: vec![0, 1],
    };
    let budget_error = ParentRecipe::prepare(
        &settings,
        &deliberately_bad_chart,
        &[],
        &dag,
        &geometry(3),
        &None,
        AffinePowerBudget::default().chart(),
        16,
    )
    .err()
    .expect("import resource limit must fail");
    assert!(
        budget_error.to_string().contains("parameter budget"),
        "{budget_error}"
    );
    let shape_error = ParentRecipe::prepare(
        &settings,
        &deliberately_bad_chart,
        &[],
        &dag,
        &geometry(3),
        &Some(vec![vec![usize::MAX, 1]]),
        AffinePowerBudget::default().chart(),
        100_000,
    )
    .err()
    .expect("analytic order must fail");
    assert!(
        shape_error.to_string().contains("analytic order"),
        "{shape_error}"
    );
}

/// Run only under the owner's finite N6/r4 preparation gate. This proves no
/// integral value and does not submit a scalar-master request.
#[test]
#[ignore = "finite controlled N6/r4 structural preparation gate"]
fn prepare_n6_rank6_affine_guard_only() {
    let mut settings = ReductionSettings::new(6, 6);
    settings.algorithm = Algorithm::Ibp;
    settings.powers[0] = 2;
    let plan = AffinePowerReductionPlan::new(settings).unwrap();
    let workspace = AffinePowerWorkspace::new(&plan, Float::new(128)).unwrap();
    assert_eq!(workspace.guard_plan.charts().len(), 1);
    assert_eq!(workspace.component_count(), 1);
    assert!(workspace.children.is_empty());
    assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls, 0);
    println!("N6 rank6 constant-one guard structural preparation completed; no integral evaluated");
}
