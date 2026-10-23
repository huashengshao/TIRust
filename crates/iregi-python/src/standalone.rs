//! One native host: shared dependencies are linked and registered exactly once.

use pyo3::prelude::*;
use symbolica::api::python::{SymbolicaCommunityModule, create_symbolica_module};

#[pymodule]
fn _tirust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    create_symbolica_module(module)?;
    let graph = PyModule::new(py, "_graph")?;
    linnet_render_py::GraphModule::register_module(&graph)?;
    let tensor = PyModule::new(py, "_tensor")?;
    spynso3::SpensoModule::register_module(&tensor)?;
    let hepkit = PyModule::new(py, "_hepkit")?;
    feynkit_py::FeynkitModule::register_module(&hepkit)?;
    crate::CommunityModule::register_module(module)?;
    // Re-export the already registered upstream class objects, retaining metadata.
    for name in ["Model", "FeynmanDiagram"] {
        module.add(name, hepkit.getattr(name)?)?;
    }
    module.add("TensorExpression", tensor.getattr("TensorExpression")?)?;
    for child in [&graph, &tensor, &hepkit] {
        module.add_submodule(child)?;
    }
    linnet_render_py::GraphModule::initialize(py)?;
    spynso3::SpensoModule::initialize(py)?;
    feynkit_py::FeynkitModule::initialize(py)?;
    crate::CommunityModule::initialize(py)?;
    module.add(
        "__all__",
        [
            "ReductionPlan",
            "Workspace",
            "KinematicsBatch",
            "BatchResult",
            "ContractionResult",
            "PreparedGraph",
            "PreparedNumerator",
            "Dual",
            "ReductionError",
            "LaneError",
            "Expression",
            "E",
            "S",
            "Float",
            "ComplexFloat",
            "Model",
            "FeynmanDiagram",
            "TensorExpression",
            "initialize",
            "configure_data_directory",
            "__version__",
        ],
    )?;
    Ok(())
}
