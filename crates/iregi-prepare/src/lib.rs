//! Graph and numerator preparation, independent of the reduction implementation.
//!
//! Symbolica owns polynomial operations; FeynKit owns graph parsing and routing;
//! Idenso and Spenso own tensor contraction and component expansion. Production
//! model input is JSON, so this crate does not embed the Python UFO loader.

use std::sync::Arc;

use feynkit_generator::FeynmanDiagramRulesExt;
use feynkit_graph::{ExternalState, FeynmanDiagram, expressions::evaluate_overall_factor, symbols};
use idenso::{CookMode, CookSettings, tensor::SymbolicTensor};
use spenso::{
    network::{
        ExecutionResult, MinIntermediateCost, Network, Sequential,
        library::{
            function_lib::{SymbolLib, Wrap},
            symbolic::{ExplicitKey, TensorLibrary},
        },
        parsing::{ParseSettings, ShadowedStructure, ShorthandParsing},
        store::NetworkStore,
    },
    structure::{
        TensorDataLayout, TensorStructure, abstract_index::AbstractIndex, partial::PartialStructure,
    },
    tensors::{data::GetTensorData, parametric::ParamTensor},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::atom::AtomField,
    function,
    id::Replacement,
    symbol,
};
use thiserror::Error;

mod model;

/// Whether DOT supplies the numerator or only a physical topology.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NumeratorSource {
    #[default]
    Supplied,
    FeynmanRules,
}

#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    pub source: NumeratorSource,
    /// Include FeynKit's symbolic external spin wavefunctions.
    pub external_wavefunctions: bool,
    /// Select residual open tensor axes in their declared logical order.
    /// Without a selection the numerator must be scalar.
    pub component: Option<Vec<usize>>,
}

/// A coefficient of q0^a q1^b q2^c q3^d, without factorial weights.
#[derive(Debug, Clone)]
pub struct PolynomialTerm {
    pub exponents: [u16; 4],
    pub coefficient: Atom,
}

#[derive(Debug, Clone)]
pub struct PreparedNumerator {
    pub variables: [Atom; 4],
    pub terms: Vec<PolynomialTerm>,
}

impl PreparedNumerator {
    pub fn rank(&self) -> usize {
        self.terms
            .iter()
            .map(|term| term.exponents.iter().map(|&e| usize::from(e)).sum())
            .max()
            .unwrap_or(0)
    }

    /// Exact reconstruction through Symbolica, useful for independent checks.
    pub fn expression(&self) -> Atom {
        self.terms
            .iter()
            .map(|term| {
                self.variables
                    .iter()
                    .zip(term.exponents)
                    .fold(term.coefficient.clone(), |acc, (variable, exponent)| {
                        acc * variable.pow(i64::from(exponent))
                    })
            })
            .sum()
    }
}

/// The propagator momentum is `loop_coefficient*q + sum(external_i*p_i)`.
#[derive(Debug, Clone)]
pub struct PropagatorRouting {
    pub edge_id: usize,
    pub particle_pdg: i64,
    pub loop_coefficient: i8,
    pub external_coefficients: Vec<i8>,
    pub mass_squared: Atom,
}

/// Stable external-state identity needed to bind momenta and wavefunctions.
#[derive(Debug, Clone)]
pub struct PreparedExternalLeg {
    pub edge_id: usize,
    pub index: usize,
    pub name: String,
    pub incoming: bool,
    pub particle_pdg: i64,
}

#[derive(Debug, Clone)]
pub struct PreparedGraph {
    pub name: String,
    pub numerator: PreparedNumerator,
    pub propagators: Vec<PropagatorRouting>,
    /// Physical edge IDs corresponding to external routing coefficient slots.
    pub external_edges: Vec<usize>,
    pub external_legs: Vec<PreparedExternalLeg>,
}

#[derive(Debug, Error)]
pub enum PrepareError {
    #[error("model input: {0}")]
    Model(String),
    #[error("graph input: {0}")]
    Graph(String),
    #[error("expected one loop, found {0}")]
    LoopCount(usize),
    #[error("tensor preparation: {0}")]
    Tensor(String),
    #[error("numerator has {rank} open tensor axes; select an explicit component")]
    OpenTensor { rank: usize },
    #[error("numerator is not a polynomial in the four loop components: {0}")]
    NonPolynomial(String),
    #[error("unsupported propagator {edge}: its momentum does not depend on the loop")]
    TreePropagator { edge: usize },
}

pub fn loop_variables() -> [Atom; 4] {
    ["iregi::q0", "iregi::q1", "iregi::q2", "iregi::q3"].map(|name| Atom::var(symbol!(name)))
}

/// Import an immutable model snapshot and a compact or serialized FeynKit DOT graph.
pub fn prepare_dot(
    model_json: &str,
    dot: &str,
    options: &PrepareOptions,
) -> Result<PreparedGraph, PrepareError> {
    // Register canonical tensor declarations before parsing user-supplied DOT.
    idenso::representations::initialize();
    let _ = (
        symbols::momentum(),
        symbols::loop_momentum(),
        symbols::external_momentum(),
    );
    let model = Arc::new(model::from_json(model_json)?);
    let diagram =
        FeynmanDiagram::from_dot(model, dot).map_err(|e| PrepareError::Graph(e.to_string()))?;
    prepare_diagram(&diagram, options)
}

pub fn prepare_diagram(
    diagram: &FeynmanDiagram,
    options: &PrepareOptions,
) -> Result<PreparedGraph, PrepareError> {
    if diagram.loop_count() != 1 {
        return Err(PrepareError::LoopCount(diagram.loop_count()));
    }
    let physical;
    let diagram = if options.source == NumeratorSource::FeynmanRules {
        physical = diagram
            .apply_feynman_rules()
            .map_err(|e| PrepareError::Graph(e.to_string()))?;
        &physical
    } else {
        diagram
    };
    let mut tensor = diagram
        .numerator()
        .multiply(diagram.numerator_prefactor())
        .and_then(|n| n.multiply(diagram.projector()))
        .map_err(|e| PrepareError::Tensor(e.to_string()))?;
    if options.external_wavefunctions {
        let wavefunctions = diagram
            .external_state_projector()
            .map_err(|e| PrepareError::Tensor(e.to_string()))?;
        tensor = tensor
            .multiply(&wavefunctions)
            .map_err(|e| PrepareError::Tensor(e.to_string()))?;
    }
    let expression = diagram.loop_momentum_basis().route_expression(
        &(evaluate_overall_factor(diagram.overall_factor().as_view()) * tensor.expression()),
    );
    let tensor = tensor
        .with_checked_expression(diagram.model().expand_couplings(&expression))
        .map_err(|e| PrepareError::Tensor(e.to_string()))?;
    let scalar = scalar_component_expression(tensor, options.component.as_deref())?;
    let variables = loop_variables();
    let replacements = (0..4).flat_map(|component| {
        ["spenso::cind", "spenso::find"].map(move |index_head| {
            Replacement::new(
                function!(
                    symbols::loop_momentum(),
                    0,
                    function!(symbol!(index_head), component)
                ),
                loop_variables()[component].clone(),
            )
        })
    });
    let scalar = scalar.replace_multiple(replacements);
    if scalar.contains_symbol(symbols::loop_momentum())
        || scalar.contains_symbol(symbols::momentum())
    {
        return Err(PrepareError::NonPolynomial(
            "unresolved routed loop tensor".into(),
        ));
    }
    let numerator = polynomial_from_expression(&scalar, variables)?;
    let basis = diagram.loop_momentum_basis();
    let mut propagators = Vec::new();
    let mut external_legs = Vec::new();
    for (edge_id, _, edge) in diagram.edges() {
        if let Some(external) = &edge.external {
            let particle = diagram
                .model()
                .particle_by_id(edge.particle)
                .map_err(|e| PrepareError::Model(e.to_string()))?;
            external_legs.push(PreparedExternalLeg {
                edge_id: edge_id.0,
                index: external.index,
                name: external.name.clone(),
                incoming: external.state == ExternalState::Incoming,
                particle_pdg: particle.pdg_code,
            });
        }
    }
    external_legs.sort_by_key(|leg| leg.index);
    for (edge_id, _, edge) in diagram
        .edges()
        .filter(|(_, _, edge)| edge.external.is_none())
    {
        let signature = &basis.edge_signatures[&edge_id];
        let loop_coefficient = signature.loops.integer_coefficients()[0] as i8;
        if loop_coefficient == 0 {
            return Err(PrepareError::TreePropagator { edge: edge_id.0 });
        }
        let particle = diagram
            .model()
            .particle_by_id(edge.particle)
            .map_err(|e| PrepareError::Model(e.to_string()))?;
        propagators.push(PropagatorRouting {
            edge_id: edge_id.0,
            particle_pdg: particle.pdg_code,
            loop_coefficient,
            external_coefficients: signature
                .external
                .integer_coefficients()
                .into_iter()
                .map(|n| n as i8)
                .collect(),
            mass_squared: particle.symbolic_mass(diagram.model()).pow(2),
        });
    }
    Ok(PreparedGraph {
        name: diagram.name().to_owned(),
        numerator,
        propagators,
        external_edges: basis.external_edges.iter().map(|edge| edge.0).collect(),
        external_legs,
    })
}

/// Extract through Symbolica and reject loop dependence hidden in coefficients.
pub fn polynomial_from_expression(
    expression: &Atom,
    variables: [Atom; 4],
) -> Result<PreparedNumerator, PrepareError> {
    for (index, variable) in variables.iter().enumerate() {
        if !matches!(variable.as_view(), AtomView::Var(_))
            || variables[..index].contains(variable)
        {
            return Err(PrepareError::NonPolynomial(
                "loop components must be four distinct plain symbols".into(),
            ));
        }
    }
    let expanded = expression.expand();
    // Every exact nonzero coefficient is retained, regardless of magnitude.
    // Statistical zero tests may classify inconclusive probes as zero.
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: None,
    };
    // Symbolica owns extraction. A wider exponent type lets us report the
    // public u16 interface limit instead of panicking in narrow conversion.
    let polynomial = expanded.to_polynomial_in_vars_with_field::<u32>(variables.as_slice(), &field);
    if polynomial.variables().len() != variables.len()
        || polynomial
            .variables()
            .iter()
            .zip(&variables)
            .any(|(found, expected)| found.to_atom() != *expected)
    {
        return Err(PrepareError::NonPolynomial(
            "polynomial variables changed during conversion".into(),
        ));
    }
    let mut terms = Vec::with_capacity(polynomial.nterms());
    for term in &polynomial {
        if variables.iter().any(|q| term.coefficient.contains(q)) {
            return Err(PrepareError::NonPolynomial(
                term.coefficient.to_canonical_string(),
            ));
        }
        let exponents: [u16; 4] = term.exponents.iter()
            .map(|&power| u16::try_from(power).map_err(|_| {
                PrepareError::NonPolynomial("loop component exponent exceeds u16::MAX".into())
            }))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| PrepareError::NonPolynomial("unexpected variable ordering".into()))?;
        terms.push(PolynomialTerm {
            exponents,
            coefficient: term.coefficient.clone(),
        });
    }
    Ok(PreparedNumerator { variables, terms })
}

type Tensor = ParamTensor<ShadowedStructure<AbstractIndex>>;
type Net = Network<NetworkStore<Tensor, Atom>, ExplicitKey<AbstractIndex>, Symbol, AbstractIndex>;

/// Contract a symbolic tensor and select a component in its logical axis order.
///
/// A scalar requires no selection. Open tensors require one coordinate per
/// declared axis; Spenso validates shape and maps logical to stored components.
/// The input tensor is owned, so preparation never mutates an upstream wrapper.
pub fn scalar_component_expression(
    tensor: SymbolicTensor<PartialStructure>,
    component: Option<&[usize]>,
) -> Result<Atom, PrepareError> {
    if let Some(component) = component {
        if component.len() != tensor.rank() {
            return Err(PrepareError::Tensor(format!(
                "component has {} coordinates, expected {}",
                component.len(),
                tensor.rank(),
            )));
        }
    }
    if component.is_none() && !tensor.is_scalar() {
        return Err(PrepareError::OpenTensor {
            rank: tensor.rank(),
        });
    }
    // An expression containing only scalar symbols needs no tensor network.
    if tensor.is_scalar() && !contains_function(tensor.expression().as_view()) {
        return Ok(tensor.into_expression());
    }
    let library: TensorLibrary<ParamTensor<ExplicitKey<AbstractIndex>>, AbstractIndex> =
        spenso_hep_lib::hep_lib_atom();
    let functions: SymbolLib<Tensor, Wrap> = Wrap::new_lib();
    let settings = ParseSettings {
        shorthand_parsing: ShorthandParsing::expand_all(),
        ..Default::default()
    };
    let cooking = CookSettings::indices().with_mode(CookMode::ReversibleEncoding);
    let lowered = tensor
        .into_network(&library, &functions, &settings, Some(&cooking))
        .map_err(|e| PrepareError::Tensor(e.to_string()))?;
    let (mut network, layout): (Net, _) = lowered.into_parts();
    network
        .execute::<Sequential, MinIntermediateCost, _, _, _>(&library, &functions)
        .map_err(|e| PrepareError::Tensor(e.to_string()))?;
    match network
        .result_tensor(&library)
        .map_err(|e| PrepareError::Tensor(e.to_string()))?
    {
        ExecutionResult::One => Ok(Atom::one()),
        ExecutionResult::Zero => Ok(Atom::Zero),
        ExecutionResult::Val(value) => {
            let shape = value
                .external_dims_iter()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PrepareError::Tensor(e.to_string()))?;
            let mapping = TensorDataLayout::from_storage_shape(&shape, &layout)
                .map_err(|e| PrepareError::Tensor(e.to_string()))?;
            let index = mapping
                .logical_expanded_to_storage_flat(component.unwrap_or(&[]))
                .map_err(|e| PrepareError::Tensor(e.to_string()))?;
            // A missing entry in a sparse tensor is an exact zero. Coordinate
            // bounds were checked by the library layout above.
            Ok(value.get_owned_linear(index.into()).unwrap_or(Atom::Zero))
        }
    }
}

fn contains_function(expression: AtomView<'_>) -> bool {
    let mut found = false;
    expression.visitor(&mut |node| {
        found |= matches!(node, AtomView::Fun(_));
        !found
    });
    found
}
