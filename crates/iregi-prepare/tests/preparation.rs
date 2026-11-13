use iregi_prepare::{
    NumeratorSource, PrepareError, PrepareOptions, loop_variables, polynomial_from_expression,
    prepare_dot, prepare_dot_with_tree_propagators, prepare_scalar_dot_with_tree_propagators,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::atom::AtomField,
    parse,
};

const SCALAR_MODEL: &str = include_str!("../../../fixtures/models/phi3.json");

#[test]
fn supplied_bubble_preserves_coefficients_and_routing() {
    let prepared = prepare_dot(
        SCALAR_MODEL,
        include_str!("../../../examples/graphs/bubble.dot"),
        &PrepareOptions::default(),
    )
    .unwrap();
    assert_eq!(prepared.propagators.len(), 2);
    // FeynKit retains all physical external-edge slots in the routing basis.
    assert_eq!(prepared.external_edges.len(), 2);
    assert_eq!(prepared.numerator.rank(), 2);
    assert_eq!(prepared.numerator.terms.len(), 6);
    assert!(
        prepared
            .propagators
            .iter()
            .all(|p| p.loop_coefficient.abs() == 1)
    );
    let expected =
        parse!("1 + 2*iregi::q0 + iregi::q0^2 - iregi::q1^2 - iregi::q2^2 - iregi::q3^2");
    assert!(
        (prepared.numerator.expression() - expected)
            .expand()
            .is_zero()
    );
}

#[test]
fn triangle_product_is_expanded_by_symbolica() {
    let prepared = prepare_dot(
        SCALAR_MODEL,
        include_str!("../../../examples/graphs/triangle.dot"),
        &PrepareOptions::default(),
    )
    .unwrap();
    assert_eq!(prepared.propagators.len(), 3);
    assert_eq!(prepared.numerator.rank(), 2);
    assert_eq!(prepared.numerator.terms.len(), 5);
    let expected = parse!("2+2*iregi::q0+iregi::q1+iregi::q0*iregi::q1+3*iregi::q2*iregi::q3");
    assert!(
        (prepared.numerator.expression() - expected)
            .expand()
            .is_zero()
    );
}

#[test]
fn nonpolynomial_loop_dependence_is_rejected() {
    for expression in [
        parse!("1/iregi::q0"),
        parse!("sin(iregi::q1)"),
        parse!("iregi::q2^(1/2)"),
    ] {
        assert!(matches!(
            polynomial_from_expression(&expression, loop_variables()),
            Err(PrepareError::NonPolynomial(_))
        ));
    }
}

#[test]
fn zero_polynomial_and_exact_symbolic_coefficients() {
    let zero = polynomial_from_expression(&Atom::Zero, loop_variables()).unwrap();
    assert!(zero.terms.is_empty());
    let expression = parse!("(a+b)/7*iregi::q0^3 + 1/13*iregi::q1");
    let prepared = polynomial_from_expression(&expression, loop_variables()).unwrap();
    assert_eq!(prepared.rank(), 3);
    assert!((prepared.expression() - expression).expand().is_zero());
}

#[test]
fn long_sums_convert_like_the_direct_symbolica_conversion() {
    // (q0 + q1 + q2 + q3 + a + ... + f)^6 expands to C(15, 6) = 5005 terms,
    // past the pairwise-conversion threshold (4096 terms).
    let expression =
        parse!("(iregi::q0+iregi::q1+iregi::q2+iregi::q3+a+b+c+d+e+f)^6").expand();
    let prepared = polynomial_from_expression(&expression, loop_variables()).unwrap();
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: None,
    };
    let direct =
        expression.to_polynomial_in_vars_with_field::<u32>(loop_variables().as_slice(), &field);
    assert_eq!(prepared.terms.len(), direct.nterms());
    for (term, reference) in prepared.terms.iter().zip(&direct) {
        assert_eq!(term.exponents.map(u32::from).as_slice(), reference.exponents);
        assert_eq!(term.coefficient, *reference.coefficient);
    }
}

#[test]
fn tiny_exact_coefficients_are_never_discarded() {
    let expression = parse!("1/10^100*(a+b)*iregi::q0^2 + 1/10^200*iregi::q1");
    let prepared = polynomial_from_expression(&expression, loop_variables()).unwrap();
    assert_eq!(prepared.terms.len(), 2);
    assert!((prepared.expression() - expression).expand().is_zero());
    assert!(
        prepared
            .terms
            .iter()
            .all(|term| !term.coefficient.is_zero())
    );
}

#[test]
fn oversized_loop_exponents_are_errors_without_integer_wraparound() {
    // Each expression is one monomial: no enormous sum is expanded.
    for exponent in ["65536", "2147483648", "4294967295", "4294967296", "18446744073709551616"] {
        let expression = Atom::parse(&format!("iregi::q0^{exponent}"), "iregi_prepare_test", Default::default()).unwrap();
        assert!(matches!(polynomial_from_expression(&expression, loop_variables()), Err(PrepareError::NonPolynomial(_))));
    }
    let expression = parse!("iregi::q0^65535");
    let prepared = polynomial_from_expression(&expression, loop_variables()).unwrap();
    assert_eq!(prepared.terms.len(), 1);
    assert_eq!(prepared.terms[0].exponents, [65535, 0, 0, 0]);
}

#[test]
fn invalid_or_duplicate_loop_variables_are_errors() {
    let expression = parse!("iregi::q0+1");
    for invalid in [Atom::one(), parse!("iregi::q0+iregi::q1"), parse!("iregi::q0^2"), parse!("iregi::q1")] {
        let mut variables = loop_variables();
        variables[0] = invalid;
        assert!(matches!(polynomial_from_expression(&expression, variables), Err(PrepareError::NonPolynomial(_))));
    }
}

#[test]
fn feynman_rules_can_be_applied_without_python() {
    let options = PrepareOptions {
        source: NumeratorSource::FeynmanRules,
        ..Default::default()
    };
    let graph = "digraph bubble { edge [particle=phi]; ext [style=invis]; ext -> a; a -> b [lmb_id=0]; a -> b; b -> ext; }";
    let prepared = prepare_dot(SCALAR_MODEL, graph, &options).unwrap();
    assert_eq!(prepared.propagators.len(), 2);
    assert_eq!(prepared.numerator.rank(), 0);
    assert!(!prepared.numerator.expression().is_zero());
}

#[test]
fn non_one_loop_graph_is_rejected() {
    let graph =
        "digraph tree { edge [particle=phi]; ext [style=invis]; ext -> a; a -> ext; a -> ext; }";
    assert!(matches!(
        prepare_dot(SCALAR_MODEL, graph, &PrepareOptions::default()),
        Err(PrepareError::LoopCount(0))
    ));
}

#[test]
fn tree_graphs_prepare_through_the_scalar_path_only() {
    // s-channel phi^3 exchange: two vertices, one internal tree propagator.
    let graph = "digraph tree { edge [particle=phi]; ext [style=invis]; ext -> a; ext -> a; a -> b; b -> ext; b -> ext; }";
    let options = PrepareOptions {
        source: NumeratorSource::FeynmanRules,
        ..Default::default()
    };
    assert!(matches!(
        prepare_dot_with_tree_propagators(SCALAR_MODEL, graph, &options),
        Err(PrepareError::LoopCount(0))
    ));
    let tree = prepare_scalar_dot_with_tree_propagators(SCALAR_MODEL, graph, &options).unwrap();
    assert!(tree.propagators.is_empty());
    assert_eq!(tree.tree_propagators.len(), 1);
    assert_eq!(tree.tree_propagators[0].loop_coefficient, 0);
    assert_eq!(tree.external_legs.len(), 4);
    assert!(!tree.scalar.is_zero());
    assert!(loop_variables().iter().all(|q| !tree.scalar.contains(q)));
    let graph = tree.into_graph().unwrap();
    assert_eq!(graph.numerator.rank(), 0);
}

#[test]
fn physical_top_triangle_has_a_nonzero_quadratic_numerator() {
    let options = PrepareOptions {
        source: NumeratorSource::FeynmanRules,
        external_wavefunctions: true,
        component: Some(vec![0, 0]),
    };
    let prepared = prepare_dot(
        include_str!("../../../fixtures/models/sm.json"),
        include_str!("../../../examples/graphs/ggh_top_triangle.dot"),
        &options,
    )
    .unwrap();
    assert_eq!(prepared.propagators.len(), 3);
    assert_eq!(prepared.numerator.rank(), 2);
    assert!(!prepared.numerator.terms.is_empty());
    assert!(
        prepared
            .propagators
            .iter()
            .all(|p| p.particle_pdg.abs() == 6)
    );
}

#[test]
fn physical_top_box_has_a_nonzero_quartic_numerator() {
    let options = PrepareOptions {
        source: NumeratorSource::FeynmanRules,
        external_wavefunctions: true,
        component: Some(vec![0, 0]),
    };
    let prepared = prepare_dot(
        include_str!("../../../fixtures/models/sm.json"),
        include_str!("../../../examples/graphs/gghh_top_box.dot"),
        &options,
    )
    .unwrap();
    assert_eq!(prepared.propagators.len(), 4);
    assert_eq!(prepared.numerator.rank(), 4);
    assert!(!prepared.numerator.terms.is_empty());
}

#[test]
fn routed_component_numerator_uses_the_selected_loop_basis() {
    let dot = r#"digraph bubble {
        graph [num="gammalooprs::K(0,spenso::cind(0))^2 + 2*gammalooprs::K(0,spenso::cind(1))"];
        edge [particle=phi]; ext [style=invis];
        ext -> a; a -> b [lmb_id=0]; a -> b; b -> ext;
    }"#;
    let prepared = prepare_dot(SCALAR_MODEL, dot, &PrepareOptions::default()).unwrap();
    let expected = parse!("iregi::q0^2+2*iregi::q1");
    assert!(
        (prepared.numerator.expression() - expected)
            .expand()
            .is_zero()
    );
}

#[test]
fn spenso_minkowski_contraction_sets_spatial_signs() {
    let dot = r#"digraph bubble {
        graph [num="spenso::dot(gammalooprs::K(0,spenso::mink(4)),gammalooprs::K(0,spenso::mink(4)))"];
        edge [particle=phi]; ext [style=invis];
        ext -> a; a -> b [lmb_id=0]; a -> b; b -> ext;
    }"#;
    let prepared = prepare_dot(SCALAR_MODEL, dot, &PrepareOptions::default()).unwrap();
    let expected = parse!("iregi::q0^2-iregi::q1^2-iregi::q2^2-iregi::q3^2");
    assert!(
        (prepared.numerator.expression() - expected)
            .expand()
            .is_zero()
    );
}

#[test]
fn missing_sparse_tensor_entry_is_zero() {
    let dot = r#"digraph bubble {
        graph [num="spenso::g(spenso::mink(4,mu),spenso::mink(4,nu))"];
        edge [particle=phi]; ext [style=invis];
        ext -> a; a -> b [lmb_id=0]; a -> b; b -> ext;
    }"#;
    let options = PrepareOptions {
        component: Some(vec![0, 1]),
        ..Default::default()
    };
    let prepared = prepare_dot(SCALAR_MODEL, dot, &options).unwrap();
    assert!(prepared.numerator.terms.is_empty());
}

#[test]
fn fixed_gluon_color_traces_are_contracted_by_spenso() {
    use idenso::representations::{ColorAdjoint, ColorFundamental};
    use spenso::structure::representation::RepName;
    use symbolica::symbol;

    idenso::representations::initialize();
    let adjoint = ColorAdjoint {}.new_rep(8);
    let fundamental = ColorFundamental {}.new_rep(3);
    for (count, expected) in [(2, parse!("1/2")), (4, parse!("1/8"))] {
        let trace = spenso::trace!(fundamental.to_symbolic([]); (0..count).map(|i| {
            let index = Atom::var(symbol!(&format!("iregi_validation::color{i}")));
            idenso::color_t!(adjoint.to_symbolic([index]))
        }));
        let dot = format!(
            "digraph color_trace {{ graph [num=\"{}\"]; edge [particle=phi]; ext [style=invis]; ext -> a; a -> b [lmb_id=0]; a -> b; b -> ext; }}",
            trace.to_canonical_string()
        );
        let prepared = prepare_dot(
            SCALAR_MODEL,
            &dot,
            &PrepareOptions {
                component: Some(vec![0; count]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(prepared.numerator.expression(), expected);
    }
}

#[test]
fn tree_propagators_are_listed_and_keep_the_loop_numerator() {
    let options = PrepareOptions {
        source: NumeratorSource::FeynmanRules,
        ..Default::default()
    };
    // The triangle with one external leg replaced by a tree propagator c -> d
    // and two external legs at d.
    let triangle = "digraph t { edge [particle=phi]; ext [style=invis]; ext -> a; b -> ext; c -> ext; a -> b [lmb_id=0]; b -> c; c -> a; }";
    let with_tree = "digraph t { edge [particle=phi]; ext [style=invis]; ext -> a; b -> ext; c -> d; d -> ext; d -> ext; a -> b [lmb_id=0]; b -> c; c -> a; }";
    let plain = prepare_dot(SCALAR_MODEL, triangle, &options).unwrap();
    // The default entry point still refuses the tree leg.
    assert!(matches!(
        prepare_dot(SCALAR_MODEL, with_tree, &options),
        Err(PrepareError::TreePropagator { .. })
    ));
    let tree = prepare_dot_with_tree_propagators(SCALAR_MODEL, with_tree, &options).unwrap();
    assert_eq!(plain.tree_propagators.len(), 0);
    assert_eq!(tree.propagators.len(), 3);
    assert_eq!(tree.tree_propagators.len(), 1);
    let leg = &tree.tree_propagators[0];
    assert_eq!(leg.loop_coefficient, 0);
    assert!(leg.external_coefficients.iter().any(|&c| c != 0), "{leg:?}");
    // The loop numerator keeps its loop dependence (rank 0 here); the tree
    // leg adds a loop-independent factor (a coupling and the propagator
    // numerator).
    assert_eq!(tree.numerator.rank(), plain.numerator.rank());
    assert!(!tree.numerator.expression().is_zero());
}
