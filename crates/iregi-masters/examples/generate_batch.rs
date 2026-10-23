//! Emit instruction-major Rust kernels from the production scalar expressions.
//!
//! From the workspace root:
//! `cargo run -p iregi-masters --no-default-features --example generate_batch`
//! The optional sole argument overrides `crates/iregi-masters/src/generated`.

use iregi_masters::{ScalarIntegral, exact_scalar_master_evaluator};
use oneloop::rust_codegen::{EmitOptions, RustEmitter};
use std::path::PathBuf;

fn generate(directory: PathBuf) -> Result<(), String> {
    let mut emitter = RustEmitter::default();
    let mut files = Vec::new();
    for (family, filename) in [
        (ScalarIntegral::A0, "a0.rs"),
        (ScalarIntegral::B0, "b0.rs"),
        (ScalarIntegral::DB0, "db0.rs"),
        (ScalarIntegral::C0, "c0.rs"),
        (ScalarIntegral::D0, "d0.rs"),
    ] {
        eprintln!("exporting scalar batch {}", family.name());
        let exact = exact_scalar_master_evaluator(family).map_err(|error| error.to_string())?;
        // Match the upstream emitter's arithmetic choices: inlining benefits
        // the small families; shared operations keep C0/D0 code size bounded.
        let inline_arithmetic = matches!(
            family,
            ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::DB0
        );
        let module = emitter.emit_batch(exact, EmitOptions { inline_arithmetic })?;
        // The private sixth output is part of the production buffer layout.
        // Scalar kernels retain it even though only Dual evaluation uses it.
        if module.input_count != family.arity() || module.output_count != 6 {
            return Err(format!(
                "{} emitted dimensions ({}, {}), expected ({}, 6)",
                family.name(),
                module.input_count,
                module.output_count,
                family.arity()
            ));
        }
        eprintln!(
            "{}: {} instructions, {} helpers, {} outlined branches, {} bytes",
            family.name(),
            module.instruction_count,
            module.helper_count,
            module.outlined_branch_count,
            module.source.len()
        );
        files.push((filename, module.source));
    }
    // All modules share one exact constant table. Generate that table only
    // after emitting every family, so every module's assigned indices agree.
    files.push(("constants.rs", emitter.constants_source()));
    // Resolve every graph before replacing any file. Unsupported instructions
    // or unresolved constants cannot leave a partial new set of modules.
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    for (filename, source) in files {
        let path = directory.join(filename);
        std::fs::write(&path, source)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    eprintln!(
        "wrote five scalar batch kernels and {} exact constants to {}",
        emitter.constants().len(),
        directory.display()
    );
    Ok(())
}

fn main() {
    let mut arguments = std::env::args_os().skip(1);
    let directory = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/generated"));
    if arguments.next().is_some() {
        eprintln!("usage: generate_batch [output-directory]");
        std::process::exit(2);
    }
    let result = std::thread::Builder::new()
        .name("iregi-master-codegen".into())
        .stack_size(128 * 1024 * 1024)
        .spawn(move || generate(directory))
        .map_err(|error| format!("start generator: {error}"))
        .and_then(|worker| {
            worker
                .join()
                .map_err(|_| "scalar batch generator panicked".to_owned())?
        });
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
