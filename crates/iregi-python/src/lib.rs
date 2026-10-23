//! TIRust bindings. Numerical reduction and input preparation remain Rust APIs.
#![forbid(unsafe_code)]

mod numerator;
mod numerical;
mod scalar;
#[cfg(feature = "standalone")]
mod standalone;

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule};
use std::path::PathBuf;
use symbolica::api::python::SymbolicaCommunityModule;

pub use numerator::{PyPreparedGraph, PyPreparedNumerator};
pub use numerical::{
    PyBatchResult, PyContractionResult, PyKinematicsBatch, PyReductionPlan, PyWorkspace,
};
pub use scalar::PyDual;

pyo3::create_exception!(_tirust, ReductionError, PyRuntimeError);
pyo3::create_exception!(_tirust, LaneError, ReductionError);

/// Link this library into either a standalone host or a Symbolica Community host.
pub struct CommunityModule;

impl SymbolicaCommunityModule for CommunityModule {
    fn get_name() -> String {
        "hepkit_tensor_integrals".into()
    }

    fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<PyReductionPlan>()?;
        module.add_class::<PyWorkspace>()?;
        module.add_class::<PyKinematicsBatch>()?;
        module.add_class::<PyBatchResult>()?;
        module.add_class::<PyContractionResult>()?;
        module.add_class::<PyDual>()?;
        module.add_class::<PyPreparedNumerator>()?;
        module.add_class::<PyPreparedGraph>()?;
        module.add("ReductionError", module.py().get_type::<ReductionError>())?;
        module.add("LaneError", module.py().get_type::<LaneError>())?;
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add_function(wrap_pyfunction!(initialize, module)?)?;
        module.add_function(wrap_pyfunction!(configure_data_directory, module)?)?;
        Ok(())
    }
}

/// Initialize this provider; safe to repeat in either native host.
#[pyfunction]
pub fn initialize(py: Python<'_>) -> PyResult<()> {
    CommunityModule::initialize(py)
}

/// Configure this standalone host's unchanged upstream rendering resources.
///
/// The Community provider does not call this automatically; its host owns data.
#[pyfunction]
#[pyo3(signature=(directory, *, validate=true))]
pub fn configure_data_directory(directory: PathBuf, validate: bool) -> PyResult<()> {
    symbolica_data::Data::set_directory(directory);
    if validate {
        symbolica_data::Asset::prepare(typst_renderer::data_assets::ALL, "render/fonts/")
            .map_err(reduction_error)?;
        symbolica_data::Asset::prepare(spynso3::data_assets::ALL, "render/fonts/")
            .map_err(reduction_error)?;
    }
    Ok(())
}

pub(crate) fn reduction_error(error: impl std::fmt::Display) -> PyErr {
    ReductionError::new_err(error.to_string())
}
