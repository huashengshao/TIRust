//! Run without a personal Symbolica key and verify the library's actual scope.
#![cfg(not(target_arch = "wasm32"))]

use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, atomic::{AtomicUsize, Ordering}},
};
use iregi_rust::{
    Convention, DirectReductionPlan, KinematicsBatch, MasterFamily, MasterProvider,
    ReductionPlan, ReductionSettings, TensorReductionPlan, TensorStore,
    TensorStoreOptions, TensorWorkspace, EmbedderOptions, Workspace,
};
use symbolica::{domains::float::{Complex, Float}, license::LicenseManager};

const CHILD_CASE: &str = "IREGI_OEM_TEST_CASE";

struct LockDirectory(PathBuf);
impl LockDirectory {
    fn new(case: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "tirust-oem-{}-{case}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir(&directory).unwrap();
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        Self(directory)
    }
}
impl Drop for LockDirectory {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

#[test]
fn oem_scope_without_a_personal_license() {
    for case in ["scope", "plans", "threads", "panic"] {
        let directory = LockDirectory::new(case);
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "oem_child", "--nocapture"])
            .env_remove("SYMBOLICA_LICENSE")
            .env("SYMBOLICA_LOCK_DIR", &directory.0)
            .env("SYMBOLICA_HIDE_BANNER", "1")
            .env(CHILD_CASE, case)
            .output().unwrap();
        assert!(output.status.success(), "OEM {case} child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    }
}

fn settings() -> ReductionSettings {
    let mut settings = ReductionSettings::new(1, 0);
    settings.convention = Convention::OneLoop;
    settings
}
fn sample() -> KinematicsBatch<f64> {
    KinematicsBatch { rows: 1, components: 1, values: [0.0, 0.0, 0.0, 0.0, 4.0, 3.0]
        .map(|v| Complex::new(v, 0.0)).to_vec() }
}
fn outside_is_restricted() {
    assert_eq!(LicenseManager::max_threads(2), 1,
        "TIRust must not globally license unrelated host Symbolica operations");
}
struct Probe { calls: Arc<AtomicUsize>, panic: bool }
impl MasterProvider for Probe {
    fn master(&self, family: MasterFamily, arguments: &[Complex<f64>]) -> Option<[Complex<f64>; 3]> {
        assert_eq!(LicenseManager::max_threads(2), 2,
            "the downstream master call must inherit TIRust's scoped license");
        assert_eq!(family, MasterFamily::Tadpole);
        assert_eq!(arguments.len(), family.arguments());
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "intentional provider panic for OEM scope-unwind test");
        // Declining exercises the real downstream oneloopmaster kernel.
        None
    }
}
fn direct_call(panic: bool) -> usize {
    outside_is_restricted();
    let plan = DirectReductionPlan::new(settings()).unwrap();
    outside_is_restricted();
    let mut workspace = plan.workspace(0.0f64).unwrap();
    outside_is_restricted();
    let calls = Arc::new(AtomicUsize::new(0));
    let store = TensorStore::with_options(TensorStoreOptions {
        masters: Some(Arc::new(Probe { calls: calls.clone(), panic })),
        ..Default::default()
    });
    workspace.set_tensor_store(Some(&store));
    let result = workspace.reduce_batch(&sample()).unwrap();
    assert_eq!(result.lane(0).unwrap()[0], Complex::new(4.0, 0.0));
    outside_is_restricted();
    let calls = calls.load(Ordering::SeqCst);
    assert!(calls > 0, "the real master provider path must be reached");
    calls
}

#[test]
fn oem_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else { return; };
    assert!(std::env::var_os("SYMBOLICA_LICENSE").is_none());
    outside_is_restricted();
    match case.as_str() {
        "scope" => { direct_call(false); }
        "plans" => {
            let plan = ReductionPlan::new(settings()).unwrap();
            outside_is_restricted();
            let mut workspace = Workspace::new(&plan, 0.0f64).unwrap();
            outside_is_restricted();
            assert!(workspace.reduce_batch(&sample()).unwrap().lane(0).is_ok());
            outside_is_restricted();
            let malformed = KinematicsBatch::<f64> { rows: 1, components: 1, values: Vec::new() };
            assert!(workspace.reduce_batch(&malformed).is_err());
            outside_is_restricted();
            let plan = TensorReductionPlan::new(settings()).unwrap();
            let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
            assert!(workspace.reduce_batch(&sample()).unwrap().lane(0).is_ok());
            outside_is_restricted();
            let plan = DirectReductionPlan::new(settings()).unwrap();
            let mut workspace = plan.workspace(Float::new(256)).unwrap();
            let input = KinematicsBatch { rows: 1, components: 1, values: [0, 0, 0, 0, 4, 3]
                .map(|v| Complex::new(Float::with_val(256, v), Float::new(256))).to_vec() };
            assert!(workspace.reduce_batch(&input).unwrap().lane(0).is_ok());
            outside_is_restricted();
        }
        "threads" => {
            let workers: Vec<_> = (0..2).map(|_| std::thread::spawn(|| direct_call(false))).collect();
            for worker in workers { assert!(worker.join().unwrap() > 0); }
        }
        "panic" => {
            let result = std::panic::catch_unwind(|| direct_call(true));
            assert!(result.is_err(), "the provider must actually unwind through the guarded reduction");
            outside_is_restricted();
            direct_call(false);
        }
        other => panic!("unknown OEM child case {other}"),
    }
    outside_is_restricted();
}
