//! One-loop (and tree) QCD diagrams of S-wave quarkonium production for
//! benchmarks/onia (physical amplitudes): FeynKit's generator on the pinned
//! SM model, one DOT file per diagram plus a manifest.
//!
//! usage: onia_diagrams MODEL_JSON OUTPUT_DIR PROCESS
//! PROCESS: `gg_QQ` (g g > c c~) or `gg_QQg` (g g > c c~ g).
//!
//! QCD only (QED order 0), Feynman gauge with ghosts. Self-energies on
//! external legs and tadpoles are vetoed (on-shell scheme: they enter through
//! wave-function renormalization, outside these bare amplitudes). b and t
//! quarks are vetoed (decoupled heavy flavours); u, d, s loops are massless.
use std::{collections::BTreeMap, error::Error, sync::Arc};

use feynkit_generator::{
    GenerationFilter, GenerationOptions, Process, SelfEnergyFilterOptions, TadpoleFilterOptions,
};
use iregi_prepare::load_model;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_json = std::fs::read_to_string(&args[1])?;
    let output = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(output)?;
    let (outgoing, qcd_tree): (Vec<&str>, usize) = match args[3].as_str() {
        "gg_QQ" => (vec!["c", "c~"], 2),
        "gg_QQg" => (vec!["c", "c~", "g"], 3),
        other => return Err(format!("unknown process {other}").into()),
    };
    let model = Arc::new(load_model(&model_json)?);
    let process = Process::new(["g", "g"], outgoing.clone()).with_filters(
        ["b", "b~", "t", "t~"].into_iter().map(Into::into).collect(),
        None,
        Vec::new(),
    );
    let mut manifest = Vec::new();
    for loops in [0usize, 1] {
        let mut orders = BTreeMap::new();
        orders.insert("QCD".to_string(), (qcd_tree + 2 * loops, Some(qcd_tree + 2 * loops)));
        orders.insert("QED".to_string(), (0, Some(0)));
        let options = GenerationOptions::default()
            .with_loop_count(loops, loops)?
            .with_graph_filter(GenerationFilter::SelfEnergy(SelfEnergyFilterOptions::default()))
            .with_graph_filter(GenerationFilter::Tadpoles(TadpoleFilterOptions::default()))
            .with_graph_filter(GenerationFilter::CouplingOrders(orders))
            .threads(8);
        let result = process.generate_diagrams(Arc::clone(&model), &options)?;
        for (index, diagram) in result.diagrams.iter().enumerate() {
            let name = format!("{}-L{loops}-{index:03}", args[3]);
            let dot = diagram.to_dot()?;
            std::fs::write(output.join(format!("{name}.dot")), &dot)?;
            let entry = json!({"name": name, "loops": loops});
            manifest.push(entry);
        }
        eprintln!("{} loops={loops}: {} diagrams", args[3], result.diagrams.len());
    }
    std::fs::write(output.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}
