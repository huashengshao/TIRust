use iregi_prepare::{
    NumeratorSource, PrepareError, PrepareOptions, loop_variables, polynomial_from_expression,
    prepare_dot,
};
use symbolica::{
    atom::{Atom, AtomCore},
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
