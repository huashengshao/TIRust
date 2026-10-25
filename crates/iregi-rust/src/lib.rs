//! Batched one-loop tensor integral reduction.
//!
//! Graph import and numerator preparation belong to `iregi-prepare`.
#![forbid(unsafe_code)]

pub mod basis;
pub mod certified_planar;
mod thin_n6;
mod thin_n7;
pub mod seven_point;
pub mod six_point;
pub mod certified_scalar;
pub mod contraction;
pub mod dispatch;
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
pub use dispatch::{
    TensorEngineExecution, TensorEngineKind, TensorReductionPlan, TensorWorkspace,
    TensorWorkspaceOptions,
};
pub use certified_planar::{CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace};
pub use certified_scalar::{
    CertificateExecutionStats, CertifiedScalarReductionPlan, CertifiedScalarWorkspace,
};
pub use invariant_certificate::ExactInvariantScalar;
pub use execution::{ExecutionProfile, ExecutionStats, Workspace, reduce_batch};
pub use plan::ReductionPlan;
pub use six_point::{SixPointReductionPlan, SixPointWorkspace, SixPointExecutionStats};
pub use seven_point::{SevenPointReductionPlan, SevenPointWorkspace, SevenPointExecutionStats};
pub use scalar::{ScalarKinematicsBatch, ScalarReductionPlan, ScalarWorkspace};
pub use rescue::{
    PrecisionRescueWorkspace, RescueScalar, RescueSettings, RescueStats, RescueStatus,
    RescueTrigger,
};
pub use types::*;

/// Scalar-master selection for explicitly configured workspaces.
pub use iregi_masters::MasterExecutionPolicy;
