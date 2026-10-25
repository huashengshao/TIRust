//! Independent validation worker; stdout is reserved for framed complex batches.
mod kernel;
use std::{fs::OpenOptions,io::Write,path::Path};
fn fresh_json(path:&Path,value:&serde_json::Value)->Result<(),String>{
    let mut file=OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut file,value).map_err(|e|e.to_string())?;
    file.write_all(b"\n").map_err(|e|e.to_string())
}
fn main()->std::process::ExitCode {
    let outcome=(||->Result<(),String>{
        let args=std::env::args().skip(1).collect::<Vec<_>>();
        if args.len()!=3 || !matches!(args[0].as_str(),"prepare"|"serve") {
            return Err("usage: parameter_oracle prepare|serve 2|3|4 fresh-artifact.json".into());
        }
        let n=args[1].parse::<usize>().map_err(|e|e.to_string())?;
        let prepared=kernel::Prepared::new(n)?;let mut artifact=prepared.artifact();
        if args[0]=="prepare" {return fresh_json(Path::new(&args[2]),&artifact);}
        let mut work=kernel::transport::Workspace::<f64>::new(&prepared)?;
        artifact["execution"]=work.ready(n);fresh_json(Path::new(&args[2]),&artifact)?;
        kernel::transport::serve(&mut work,n,&mut std::io::stdin().lock(),&mut std::io::stdout().lock())
    })();
    match outcome {Ok(())=>std::process::ExitCode::SUCCESS,Err(e)=>{
        eprintln!("parameter_oracle: {e}");std::process::ExitCode::FAILURE}}
}
