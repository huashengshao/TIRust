//! Direct upstream number conversion and transport-only Dual metadata.

use std::collections::HashSet;

use iregi_masters::NativeFloat;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyComplex, PyComplexMethods, PyFloat, PyInt, PyString, PyTuple},
};
use symbolica::{
    api::python::PythonExpression,
    atom::Atom,
    domains::{
        dual::HyperDual,
        float::{
            Complex, DoubleFloat, Float, FloatLike, PythonComplexFloat, PythonFloat, RealLike,
            SingleFloat,
        },
        rational::Rational,
    },
};

/// An immutable transport for Symbolica Taylor coefficients, with no arithmetic.
#[pyclass(
    frozen,
    name = "Dual",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyDual {
    pub(crate) shape: Vec<Vec<usize>>,
    pub(crate) values: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyDual {
    #[new]
    fn new(shape: Vec<Vec<usize>>, values: Vec<Py<PyAny>>) -> PyResult<Self> {
        checked_shape(&shape)?;
        if values.len() != shape.len() {
            return Err(PyValueError::new_err(
                "Dual values must match the shape component count",
            ));
        }
        Ok(Self { shape, values })
    }

    #[getter]
    fn shape(&self) -> Vec<Vec<usize>> {
        self.shape.clone()
    }

    #[getter]
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.values.iter().map(|value| value.clone_ref(py)))?.unbind())
    }

    #[getter]
    fn primal(&self, py: Python<'_>) -> Py<PyAny> {
        self.values[0].clone_ref(py)
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dual(shape={:?}, components={})",
            self.shape,
            self.values.len()
        )
    }
}

/// Validate untrusted metadata before Symbolica's infallible constructor.
/// These are index/shape checks only; Symbolica creates all multiplication data.
pub(crate) fn checked_shape(shape: &[Vec<usize>]) -> PyResult<HyperDual<u8>> {
    let fail = || {
        PyValueError::new_err(
            "Dual shape must have a leading primal, unique equally sized multi-indices, and every ancestor",
        )
    };
    let Some(first) = shape.first() else {
        return Err(fail());
    };
    if first.iter().any(|&power| power != 0) {
        return Err(fail());
    }
    let mut seen = HashSet::with_capacity(shape.len());
    for index in shape {
        if index.len() != first.len() || !seen.insert(index.as_slice()) {
            return Err(fail());
        }
        let mut degree = 0usize;
        for &power in index {
            degree = degree.checked_add(power).ok_or_else(fail)?;
            power.checked_add(power).ok_or_else(fail)?;
        }
        if degree >= shape.len() {
            return Err(fail());
        }
    }
    for index in shape {
        for axis in 0..index.len() {
            if index[axis] == 0 {
                continue;
            }
            let mut ancestor = index.clone();
            ancestor[axis] -= 1;
            if !seen.contains(ancestor.as_slice()) {
                return Err(fail());
            }
        }
    }
    let dual = HyperDual::<u8>::new(shape.to_vec());
    iregi_masters::validate_dual_structure(&dual).map_err(crate::reduction_error)?;
    Ok(dual)
}

pub(crate) trait PythonNumber: NativeFloat {
    fn from_float(value: &Float, prototype: &Self) -> PyResult<Self>;
    fn output(value: &Complex<Self>, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

impl PythonNumber for f64 {
    fn from_float(value: &Float, _: &Self) -> PyResult<Self> {
        Ok(value.to_f64())
    }
    fn output(value: &Complex<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyComplex::from_doubles(py, value.re, value.im)
            .into_any()
            .unbind())
    }
}

impl PythonNumber for DoubleFloat {
    fn from_float(value: &Float, prototype: &Self) -> PyResult<Self> {
        if value.is_zero() {
            return Ok(if value.is_sign_negative() {
                -prototype.zero()
            } else {
                prototype.zero()
            });
        }
        // Reuse Numerica's source-precision residual conversion. In contrast,
        // DoubleFloat::from_rational first rounds to 106 bits and loses wide-gap
        // tails that an existing two-component DoubleFloat can represent.
        Ok(value.to_double_float())
    }
    fn output(value: &Complex<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(
            py,
            PythonComplexFloat(Complex::new(
                exact_double_float(&value.re)?,
                exact_double_float(&value.im)?,
            )),
        )?
        .into_any())
    }
}

/// Export the exact represented pair, including tails below the nominal 106 bits.
/// Existing Symbolica rational/Float conversions perform all number arithmetic.
fn exact_double_float(value: &DoubleFloat) -> PyResult<Float> {
    if !value.is_finite() {
        // No finite precision can be lost for NaN/infinity. Preserve these in
        // prepared coefficient batches so the core can reject only that lane.
        return Ok(Float::from(*value));
    }
    let exact = value.exact_rational().ok_or_else(|| {
        PyValueError::new_err("cannot export nonfinite DoubleFloat from a successful lane")
    })?;
    if value.is_zero() {
        let zero = Float::new(106);
        return Ok(if value.negative_sign() { -zero } else { zero });
    }
    let bits = u32::try_from(exact.numerator_ref().significant_bits())
        .map_err(|_| PyValueError::new_err("DoubleFloat representation precision overflow"))?
        .max(106);
    let result = Float::new(bits).from_rational(&exact);
    if result.try_to_rational().as_ref() != Some(&exact) {
        return Err(PyValueError::new_err(
            "DoubleFloat exact export did not round-trip",
        ));
    }
    Ok(result)
}

impl PythonNumber for Float {
    fn from_float(value: &Float, _: &Self) -> PyResult<Self> {
        // Existing upstream Float values keep both value and working precision.
        Ok(value.clone())
    }
    fn output(value: &Complex<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PythonComplexFloat(value.clone()))?.into_any())
    }
}

pub(crate) fn number<T: PythonNumber>(
    value: &Bound<'_, PyAny>,
    prototype: &T,
) -> PyResult<Complex<T>> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "boolean is not a numerical input; use integer 0 or 1",
        ));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PythonComplexFloat>>() {
        return Ok(Complex::new(
            T::from_float(&value.0.re, prototype)?,
            T::from_float(&value.0.im, prototype)?,
        ));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PythonFloat>>() {
        return Ok(Complex::new(
            T::from_float(&value.0, prototype)?,
            prototype.zero(),
        ));
    }
    if let Ok(expression) = value.extract::<PyRef<'_, PythonExpression>>() {
        if let Ok(exact) = Complex::<Rational>::try_from(&expression.expr) {
            return Ok(Complex::new(
                prototype.from_rational(&exact.re),
                prototype.from_rational(&exact.im),
            ));
        }
        if let Ok(value) = Complex::<Float>::try_from(&expression.expr) {
            return Ok(Complex::new(
                T::from_float(&value.re, prototype)?,
                T::from_float(&value.im, prototype)?,
            ));
        }
        return Err(PyTypeError::new_err(
            "input Expression must be an actual numerical constant",
        ));
    }
    if let Ok(value) = value.cast::<PyComplex>() {
        return Ok(Complex::new(
            T::from_float(&Float::from(value.real()), prototype)?,
            T::from_float(&Float::from(value.imag()), prototype)?,
        ));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Complex::new(
            T::from_float(&Float::from(value.extract::<f64>()?), prototype)?,
            prototype.zero(),
        ));
    }
    if value.is_instance_of::<PyInt>() {
        let text = value.str()?.to_str()?.to_owned();
        return exact_literal(&text, prototype);
    }
    if let Ok(value) = value.cast::<PyString>() {
        let text = value.to_str()?;
        if text.contains('/') {
            return exact_literal(text, prototype);
        }
        let parsed =
            Float::parse(text, Some(prototype.get_precision())).map_err(PyValueError::new_err)?;
        return Ok(Complex::new(
            T::from_float(&parsed, prototype)?,
            prototype.zero(),
        ));
    }
    if let Ok(pair) = value.cast::<PyTuple>() {
        if pair.len() == 2 {
            let re = number(&pair.get_item(0)?, prototype)?;
            let im = number(&pair.get_item(1)?, prototype)?;
            if !re.im.is_zero() || !im.im.is_zero() {
                return Err(PyTypeError::new_err(
                    "a (real, imaginary) input pair requires real scalar components",
                ));
            }
            return Ok(Complex::new(re.re, im.re));
        }
    }
    Err(PyTypeError::new_err(
        "expected int, float, complex, decimal/rational string, numerical Expression, Float, ComplexFloat, or (real, imaginary) tuple from this native host",
    ))
}

fn exact_literal<T: PythonNumber>(text: &str, prototype: &T) -> PyResult<Complex<T>> {
    let expression = Atom::parse(text, "tirust_input", Default::default())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let exact = Complex::<Rational>::try_from(&expression).map_err(PyValueError::new_err)?;
    Ok(Complex::new(
        prototype.from_rational(&exact.re),
        prototype.from_rational(&exact.im),
    ))
}

pub(crate) fn components<T: PythonNumber>(
    value: &Bound<'_, PyAny>,
    prototype: &T,
    shape: Option<&[Vec<usize>]>,
) -> PyResult<Vec<Complex<T>>> {
    if let Ok(dual) = value.extract::<PyRef<'_, PyDual>>() {
        if shape != Some(dual.shape.as_slice()) {
            return Err(PyValueError::new_err(
                "input Dual shape differs from the workspace's exact component order",
            ));
        }
        return dual
            .values
            .iter()
            .map(|item| number(item.bind(value.py()), prototype))
            .collect();
    }
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut values = vec![zero; shape.map_or(1, <[Vec<usize>]>::len)];
    values[0] = number(value, prototype)?;
    Ok(values)
}

pub(crate) fn output_components<T: PythonNumber>(
    values: &[Complex<T>],
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    if let Some(shape) = shape {
        let values = values
            .iter()
            .map(|value| T::output(value, py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Py::new(
            py,
            PyDual {
                shape: shape.to_vec(),
                values,
            },
        )?
        .into_any())
    } else {
        T::output(&values[0], py)
    }
}

pub(crate) fn objects(value: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    value
        .try_iter()?
        .map(|item| item.map(Bound::unbind))
        .collect()
}

pub(crate) fn rows(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    value.try_iter()?.map(|row| objects(&row?)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_float_export_retains_wide_gap_cancellation_and_subnormal() {
        let cases = [
            DoubleFloat::from_compensated_sum(1., 2f64.powi(-200)),
            DoubleFloat::from_compensated_sum(1., -1. + f64::EPSILON),
            DoubleFloat::from_compensated_sum(f64::MIN_POSITIVE, f64::from_bits(1)),
            DoubleFloat::from_compensated_sum(-1., -2f64.powi(-200)),
        ];
        assert_ne!(
            cases[0].exact_rational(),
            DoubleFloat::from(1.).exact_rational()
        );
        for value in cases {
            let exported = exact_double_float(&value).unwrap();
            assert_eq!(exported.try_to_rational(), value.exact_rational());
            assert_eq!(
                DoubleFloat::from_float(&exported, &DoubleFloat::from(0.))
                    .unwrap()
                    .exact_rational(),
                value.exact_rational()
            );
        }
        assert!(
            exact_double_float(&-DoubleFloat::from(0.))
                .unwrap()
                .is_sign_negative()
        );
    }

    #[test]
    fn invalid_shapes_fail_before_upstream_construction() {
        for shape in [
            vec![],
            vec![vec![1]],
            vec![vec![0], vec![2]],
            vec![vec![0], vec![0]],
            vec![vec![0], vec![1, 0]],
        ] {
            assert!(checked_shape(&shape).is_err());
        }
        assert!(checked_shape(&[vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]).is_ok());
    }
}
