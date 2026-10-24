//! Batched one-loop tensor integral reduction.
//!
//! Graph import and numerator preparation belong to `iregi-prepare`.
#![forbid(unsafe_code)]

pub mod basis;
pub mod certified_planar;
mod thin_n6;
pub mod six_point;
pub mod certified_scalar;
pub mod contraction;
mod convention;
pub mod execution;
mod exact_zero_certificate;
pub mod ibp;
mod invariant_certificate;
pub mod plan;
pub mod pv;
pub mod rescue;
pub mod scalar;
pub mod snapping;
mod stability;
pub mod types;
pub use contraction::{ContractionPlan, ContractionResult, ContractionWorkspace, NumeratorBatch};
pub use certified_planar::{CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace};
pub use certified_scalar::{
    CertificateExecutionStats, CertifiedScalarReductionPlan, CertifiedScalarWorkspace,
};
pub use invariant_certificate::ExactInvariantScalar;
pub use execution::{ExecutionProfile, ExecutionStats, Workspace, reduce_batch};
pub use plan::ReductionPlan;
pub use six_point::{SixPointReductionPlan, SixPointWorkspace, SixPointExecutionStats};
pub use scalar::{ScalarKinematicsBatch, ScalarReductionPlan, ScalarWorkspace};
pub use rescue::{
    PrecisionRescueWorkspace, RescueScalar, RescueSettings, RescueStats, RescueStatus,
    RescueTrigger,
};
pub use types::*;
