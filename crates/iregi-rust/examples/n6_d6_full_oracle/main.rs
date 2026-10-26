#![recursion_limit = "256"]
//! Optional source-only full210 D6 weighted validation worker; no production imports.
mod kernel;
mod program;
mod stream;
pub type Answer<T>=Result<T,String>;
use std::{fs::OpenOptions,io::Write,path::Path};
fn fresh(path:&Path,value:&serde_json::Value)->Answer<()> {
    let mut output=OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut output,value).map_err(|e|e.to_string())?;
    output.write_all(b"\n").map_err(|e|e.to_string())
}
fn main()->std::process::ExitCode {
    let result=(||->Answer<()> {
        let args=std::env::args().skip(1).collect::<Vec<_>>();
        if args.len()!=3||!matches!(args[0].as_str(),"prepare"|"serve")||args[1]!="6" {
            return Err("usage: n6_d6_full_oracle prepare|serve 6 fresh-artifact.json".into());}
        let output=Path::new(&args[2]);if output.exists(){return Err("preserve existing full-oracle artifact".into());}
        let prepared=kernel::Prepared::new()?;
        let mut worker=stream::Worker::new(&prepared)?;
        let ready=worker.ready()?;
        let mut artifact=prepared.artifact();artifact["constant_values"]=ready["constant_values"].clone();artifact["execution"]=ready;
        fresh(output,&artifact)?;
        if args[0]=="prepare" {return Ok(());}
        worker.serve(&mut std::io::stdin().lock(),&mut std::io::stdout().lock())
    })();
    match result {Ok(())=>std::process::ExitCode::SUCCESS,Err(error)=>{
        eprintln!("n6_d6_full_oracle: {error}");std::process::ExitCode::FAILURE}}
}
