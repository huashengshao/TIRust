//! Batched one-loop tensor integral reduction.
//!
//! Graph import and numerator preparation belong to `iregi-prepare`.
#![forbid(unsafe_code)]

pub mod basis;
pub mod contraction;
mod convention;
pub mod execution;
pub mod ibp;
pub mod plan;
pub mod pv;
pub mod rescue;
mod stability;
pub mod types;
pub use contraction::{ContractionPlan, ContractionResult, ContractionWorkspace, NumeratorBatch};
pub use execution::{ExecutionProfile, ExecutionStats, Workspace, reduce_batch};
pub use plan::ReductionPlan;
pub use rescue::{
    PrecisionRescueWorkspace, RescueScalar, RescueSettings, RescueStats, RescueStatus,
    RescueTrigger,
};
pub use types::*;
