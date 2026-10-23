//! Actual upstream graph/tensor/Expression interoperability, with no serialization.

use std::sync::Arc;

use feynkit_graph::FeynmanDiagram;
use feynkit_py::{PyFeynmanDiagram, PyModel};
use iregi_prepare::{NumeratorSource, PrepareOptions, PreparedGraph, PreparedNumerator};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyTuple},
};
use spynso3::expression::TensorExpression;
use symbolica::{
    api::python::PythonExpression,
    atom::{Atom, AtomCore, AtomView},
    domains::{
        float::{Complex, DoubleFloat, Float},
        rational::Rational,
    },
    evaluate::{Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::{
    reduction_error,
    scalar::{PythonNumber, checked_shape, components, output_components},
};

#[pyclass(
    frozen,
    name = "PreparedNumerator",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyPreparedNumerator {
    inner: PreparedNumerator,
}

#[pymethods]
impl PyPreparedNumerator {
    /// Extract a polynomial through Symbolica; coefficients remain native Atoms.
    #[staticmethod]
    fn from_expression(
        py: Python<'_>,
        expression: PyRef<'_, PythonExpression>,
        variables: Vec<Py<PythonExpression>>,
    ) -> PyResult<Self> {
        let variables = loop_variables(py, variables)?;
        let inner = iregi_prepare::polynomial_from_expression(&expression.expr, variables)
            .map_err(reduction_error)?;
        Ok(Self { inner })
    }

    /// Contract/select actual Spenso tensor components before polynomial extraction.
    #[staticmethod]
    #[pyo3(signature = (numerator, variables, *, component=None))]
    fn from_tensor(
        py: Python<'_>,
        numerator: PyRef<'_, TensorExpression>,
        variables: Vec<Py<PythonExpression>>,
        component: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let expression = iregi_prepare::scalar_component_expression(
            numerator.structured().clone(),
            component.as_deref(),
        )
        .map_err(reduction_error)?;
        let inner =
            iregi_prepare::polynomial_from_expression(&expression, loop_variables(py, variables)?)
                .map_err(reduction_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    #[getter]
    fn expression(&self) -> PythonExpression {
        self.inner.expression().into()
    }
    #[getter]
    fn variables(&self) -> Vec<PythonExpression> {
        self.inner
            .variables
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }
    #[getter]
    fn monomials(&self) -> Vec<(u16, u16, u16, u16)> {
        self.inner
            .terms
            .iter()
            .map(|term| {
                let [a, b, c, d] = term.exponents;
                (a, b, c, d)
            })
            .collect()
    }
    #[getter]
    fn terms(&self) -> Vec<((u16, u16, u16, u16), PythonExpression)> {
        self.inner
            .terms
            .iter()
            .map(|term| {
                let [a, b, c, d] = term.exponents;
                ((a, b, c, d), term.coefficient.clone().into())
            })
            .collect()
    }

    /// Evaluate all coefficient expressions over all sample lanes in Symbolica.
    /// Rows remain explicit when parameters=[], including constant polynomials.
    #[pyo3(signature = (parameters, rows, *, dtype="f64", precision=None, dual_shape=None))]
    fn evaluate_batch(
        &self,
        py: Python<'_>,
        parameters: Vec<Py<PythonExpression>>,
        rows: &Bound<'_, PyAny>,
        dtype: &str,
        precision: Option<u32>,
        dual_shape: Option<Vec<Vec<usize>>>,
    ) -> PyResult<Py<PyAny>> {
        let parameters: Vec<_> = parameters
            .iter()
            .map(|parameter| parameter.borrow(py).expr.clone())
            .collect();
        let input = crate::scalar::rows(rows)?;
        if let Some(shape) = &dual_shape {
            checked_shape(shape)?;
        }
        match dtype {
            "f64" => {
                fixed_precision(precision, 53)?;
                evaluate(
                    &self.inner,
                    &parameters,
                    &input,
                    0.,
                    dual_shape.as_deref(),
                    py,
                )
            }
            "double-double" => {
                fixed_precision(precision, 106)?;
                evaluate(
                    &self.inner,
                    &parameters,
                    &input,
                    DoubleFloat::from(0.),
                    dual_shape.as_deref(),
                    py,
                )
            }
            "float" => {
                let prototype = Float::parse("0", Some(precision.unwrap_or(128)))
                    .map_err(PyValueError::new_err)?;
                evaluate(
                    &self.inner,
                    &parameters,
                    &input,
                    prototype,
                    dual_shape.as_deref(),
                    py,
                )
            }
            _ => Err(PyValueError::new_err(
                "dtype must be 'f64', 'double-double', or 'float'",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedNumerator(rank={}, terms={})",
            self.inner.rank(),
            self.inner.terms.len()
        )
    }
}

fn fixed_precision(precision: Option<u32>, expected: u32) -> PyResult<()> {
    if precision.is_some_and(|bits| bits != expected) {
        return Err(PyValueError::new_err(format!(
            "selected fixed type has {expected} arithmetic bits"
        )));
    }
    Ok(())
}

fn loop_variables(py: Python<'_>, variables: Vec<Py<PythonExpression>>) -> PyResult<[Atom; 4]> {
    let variables: [Atom; 4] = variables
        .into_iter()
        .map(|value| value.borrow(py).expr.clone())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| {
            PyValueError::new_err(
                "exactly four loop component Expressions are required, time component first",
            )
        })?;
    for (index, variable) in variables.iter().enumerate() {
        if !matches!(variable.as_view(), AtomView::Var(_)) {
            return Err(PyValueError::new_err(
                "loop component variables must be actual plain Symbolica symbols",
            ));
        }
        if variables[..index].contains(variable) {
            return Err(PyValueError::new_err(
                "the four loop component symbols must be distinct",
            ));
        }
    }
    Ok(variables)
}

fn evaluate<T: PythonNumber>(
    numerator: &PreparedNumerator,
    parameters: &[Atom],
    input: &[Vec<Py<PyAny>>],
    prototype: T,
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>>
where
    Complex<T>: EvaluationDomain,
{
    let component_count = shape.map_or(1, <[Vec<usize>]>::len);
    let mut values = Vec::new();
    for (lane, row) in input.iter().enumerate() {
        if row.len() != parameters.len() {
            return Err(PyValueError::new_err(format!(
                "lane {lane}: coefficient parameter count differs from declared parameters"
            )));
        }
        for value in row {
            values.extend(components(value.bind(py), &prototype, shape)?);
        }
    }
    let expressions: Vec<_> = numerator
        .terms
        .iter()
        .map(|term| term.coefficient.clone())
        .collect();
    let size = input
        .len()
        .checked_mul(expressions.len())
        .and_then(|size| size.checked_mul(component_count))
        .ok_or_else(|| PyValueError::new_err("coefficient batch dimensions overflow"))?;
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut output = vec![zero; size];
    if !expressions.is_empty() {
        let mut exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&expressions, parameters)
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .map_err(reduction_error)?;
        if let Some(shape) = shape {
            exact = exact
                .vectorize(&Dualizer::new(checked_shape(shape)?, Vec::new()))
                .map_err(reduction_error)?;
        } else {
            exact.optimize_stack();
        }
        let mut evaluator = exact
            .map_coeff_with_prec(
                &|coefficient| {
                    Complex::new(
                        prototype.from_rational(&coefficient.re),
                        prototype.from_rational(&coefficient.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator();
        let row_count = input.len();
        py.detach(|| evaluator.evaluate_batch(&values, &mut output, row_count))
            .map_err(reduction_error)?;
    }
    let mut lanes = Vec::with_capacity(input.len());
    // Explicit indexing retains lane count for a zero polynomial (zero columns).
    let lane_width = expressions.len() * component_count;
    for lane in 0..input.len() {
        let coefficients = (0..expressions.len())
            .map(|term| {
                let start = lane * lane_width + term * component_count;
                output_components(&output[start..start + component_count], shape, py)
            })
            .collect::<PyResult<Vec<_>>>()?;
        lanes.push(PyTuple::new(py, coefficients)?.into_any().unbind());
    }
    Ok(PyTuple::new(py, lanes)?.into_any().unbind())
}

#[pyclass(
    frozen,
    name = "PreparedGraph",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyPreparedGraph {
    inner: PreparedGraph,
    diagram: Py<PyFeynmanDiagram>,
}

#[pymethods]
impl PyPreparedGraph {
    /// Prepare the actual native diagram and retain its Python object identity.
    #[staticmethod]
    #[pyo3(signature = (diagram, *, source="supplied", external_wavefunctions=false, component=None))]
    fn from_diagram(
        py: Python<'_>,
        diagram: Py<PyFeynmanDiagram>,
        source: &str,
        external_wavefunctions: bool,
        component: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let inner = {
            let borrowed = diagram.borrow(py);
            iregi_prepare::prepare_diagram(
                borrowed.as_diagram()?,
                &options(source, external_wavefunctions, component)?,
            )
            .map_err(reduction_error)?
        };
        Ok(Self { inner, diagram })
    }

    /// Import FeynKit DOT with the actual shared native Model, without JSON handoff.
    #[staticmethod]
    #[pyo3(signature = (model, dot, *, source="supplied", external_wavefunctions=false, component=None))]
    fn from_dot(
        py: Python<'_>,
        model: PyRef<'_, PyModel>,
        dot: &str,
        source: &str,
        external_wavefunctions: bool,
        component: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let diagram =
            FeynmanDiagram::from_dot(Arc::clone(model.as_model()), dot).map_err(reduction_error)?;
        let inner = iregi_prepare::prepare_diagram(
            &diagram,
            &options(source, external_wavefunctions, component)?,
        )
        .map_err(reduction_error)?;
        Ok(Self {
            inner,
            diagram: Py::new(py, PyFeynmanDiagram::from(diagram))?,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn propagator_count(&self) -> usize {
        self.inner.propagators.len()
    }
    #[getter]
    fn diagram(&self, py: Python<'_>) -> Py<PyFeynmanDiagram> {
        self.diagram.clone_ref(py)
    }
    #[getter]
    fn numerator(&self) -> PyPreparedNumerator {
        PyPreparedNumerator {
            inner: self.inner.numerator.clone(),
        }
    }
    #[getter]
    fn external_edges(&self) -> Vec<usize> {
        self.inner.external_edges.clone()
    }
    #[getter]
    fn propagators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .propagators
            .iter()
            .map(|propagator| {
                let value = PyDict::new(py);
                value.set_item("edge_id", propagator.edge_id)?;
                value.set_item("particle_pdg", propagator.particle_pdg)?;
                value.set_item("loop_coefficient", propagator.loop_coefficient)?;
                value.set_item(
                    "external_coefficients",
                    propagator.external_coefficients.clone(),
                )?;
                value.set_item(
                    "mass_squared",
                    Py::new(py, PythonExpression::from(propagator.mass_squared.clone()))?,
                )?;
                Ok(value.unbind())
            })
            .collect()
    }
    fn __repr__(&self) -> String {
        format!(
            "PreparedGraph(name={:?}, propagators={}, rank={})",
            self.inner.name,
            self.inner.propagators.len(),
            self.inner.numerator.rank()
        )
    }
}

fn options(
    source: &str,
    external_wavefunctions: bool,
    component: Option<Vec<usize>>,
) -> PyResult<PrepareOptions> {
    let source = match source {
        "supplied" => NumeratorSource::Supplied,
        "feynman_rules" => NumeratorSource::FeynmanRules,
        _ => {
            return Err(PyValueError::new_err(
                "source must be 'supplied' or 'feynman_rules'",
            ));
        }
    };
    Ok(PrepareOptions {
        source,
        external_wavefunctions,
        component,
    })
}
