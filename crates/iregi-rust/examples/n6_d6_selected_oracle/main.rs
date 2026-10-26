//! Optional independent validation worker. Stdout is complex-v2 frames only.
//! Source proposal; the owner separately authorizes build/tests/integration.
mod selected_kernel;
mod transport;
pub use selected_kernel::{Answer, SelectedPrepared as Prepared};
use std::{fs::OpenOptions, io::Write, path::Path};

fn fresh_json(path: &Path, value: &serde_json::Value) -> Answer<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut file,value).map_err(|error|error.to_string())?;
    file.write_all(b"\n").map_err(|error|error.to_string())
}

fn main() -> std::process::ExitCode {
    let outcome = (|| -> Answer<()> {
        let args=std::env::args().skip(1).collect::<Vec<_>>();
        if args.len()!=3 || !matches!(args[0].as_str(),"prepare"|"serve") || args[1]!="6" {
            return Err("usage: n6_d6_selected_oracle prepare|serve 6 fresh-artifact.json".into());
        }
        let output=Path::new(&args[2]);
        if output.exists() {return Err("preserve existing selected-oracle artifact".into());}
        let prepared=Prepared::new()?;
        let mut artifact=prepared.artifact();
        if args[0]=="prepare" {return fresh_json(output,&artifact);}
        let mut work=transport::Workspace::<f64>::new(&prepared)?;
        artifact["execution"]=work.ready(6);
        fresh_json(output,&artifact)?;
        transport::serve(&mut work,6,&mut std::io::stdin().lock(),&mut std::io::stdout().lock())
    })();
    match outcome {
        Ok(())=>std::process::ExitCode::SUCCESS,
        Err(error)=>{
            eprintln!("n6_d6_selected_oracle: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
