//! Full210 native D6 kernels, obtained solely through existing Symbolica source operations.
#[path="kernel_prefix.rs"] mod existing;
use crate::{Answer,program::Program};
use serde_json::{Value,json};
use std::collections::HashMap;
use symbolica::{atom::{Atom,AtomCore,Symbol},domains::rational::Rational};
use sha2::{Digest,Sha256};

pub const MOMENTS:usize=210;
pub const OUTPUTS:usize=840;
pub struct Prepared {
    pub basis:Vec<[u16;4]>,pub map:Vec<[usize;2]>,pub expressions:Vec<Atom>,
    pub dynamic_expressions:Vec<Atom>,pub constant_expressions:Vec<Atom>,
    pub dynamic:Program,pub constants:Option<Program>,pub geometry:[Program;2],pub map_digest:String,
}
impl Prepared {
    pub fn new()->Answer<Self> {
        let s=existing::Symbols::new();
        let f_symbol=symbolica::symbol!("low_n_parameter_oracle::F");
        if Atom::var(f_symbol)!=s.f {return Err("F source identity differs".into());}
        let [uv,finite]=existing::laurent_kernels(4,&s,true)?;
        let (uv,_)=existing::source_coefficients(&uv,&s)?;
        let (finite,_)=existing::source_coefficients(&finite,&s)?;
        if uv.len()!=330 || finite.len()!=330 {return Err("incomplete original N4 source tables".into());}
        let basis=existing::basis().into_iter().take(MOMENTS).collect::<Vec<_>>();
        if basis.len()!=MOMENTS || basis.iter().any(|a|a.iter().sum::<u16>()>6) {return Err("rank6 basis differs".into());}
        let factor=-Atom::var(Symbol::PI);
        let mut expressions=Vec::with_capacity(OUTPUTS);
        for (index,alpha) in basis.iter().enumerate() {
            let uv6=(&factor*uv[index].derivative(f_symbol)).expand().together();
            let finite6=(&factor*finite[index].derivative(f_symbol)).expand().together();
            let uv_expected=alpha.iter().sum::<u16>()==6 && alpha.iter().all(|power|power%2==0);
            if (!uv6.is_zero())!=uv_expected {return Err("D6 UV support differs from degree6 all-even monomials".into());}
            expressions.extend([uv6,Atom::new(),Atom::new(),finite6]);
        }
        let parameters=s.parameters();
        let mut dynamic_expressions=Vec::new();let mut constant_expressions=Vec::new();
        let mut dynamic_index=HashMap::<Atom,usize>::new();let mut constant_index=HashMap::<Atom,usize>::new();
        let mut map=Vec::with_capacity(OUTPUTS);let mut uv_nonzero=0;
        for (output,expression) in expressions.iter().enumerate() {
            if expression.is_zero() {map.push([0,0]);continue;}
            let dynamic=parameters.iter().any(|p|expression.contains(p));
            if output%4==1 || output%4==2 {return Err("positive-F IR source is nonzero".into());}
            if output%4==0 {uv_nonzero+=1;if dynamic {return Err("rank6 D6 UV source unexpectedly parameter-dependent".into());}}
            let (table,indices,kind)=if dynamic {(&mut dynamic_expressions,&mut dynamic_index,1)}
                else {(&mut constant_expressions,&mut constant_index,2)};
            let index=*indices.entry(expression.clone()).or_insert_with(||{let i=table.len();table.push(expression.clone());i});
            map.push([kind,index]);
        }
        if uv_nonzero!=20 || dynamic_expressions.is_empty() || dynamic_expressions.len()>210 || constant_expressions.len()>20 {
            return Err("Symbolica zero/constant/dynamic inventory differs from full D6 rank6 domain".into());
        }
        let dynamic=Program::new(&dynamic_expressions,&parameters)?;
        if dynamic.inventory.slots.checked_mul(256).ok_or("slot count overflow")?>crate::program::MAX_SLOT_VALUES {
            return Err("dynamic evaluator does not fit fixed256row numeric-slot budget".into());
        }
        let constants=if constant_expressions.is_empty(){None}else{Some(Program::new(&constant_expressions,&parameters)?)};
        let map_digest=format!("{:x}",Sha256::digest(serde_json::to_vec(&map).map_err(|e|e.to_string())?));
        let geometry=[geometry_program(false)?,geometry_program(true)?];
        Ok(Self{basis,map,expressions,dynamic_expressions,constant_expressions,dynamic,constants,geometry,map_digest})
    }
    pub fn artifact(&self)->Value {
        let d=self.dynamic_expressions.len();let c=self.constant_expressions.len();
        json!({"status":"full210_D6_Symbolica_prepared_not_integrated","propagators":6,"dimension_shift_h":1,
            "moments":MOMENTS,"native_slots":4,"outputs":OUTPUTS,"basis":self.basis,
            "powers":[1,1,1,1,1,1],"physical_scale_mu":"3","physical_log":"log(9)-log(pi)-log(F)",
            "native_normalization_included":true,"native_phase":"i*pi^3","raw_unweighted_components":true,
            "ir_poles_proven_zero":true,"nonzero_degree6_UV_components":20,
            "dynamic_count":d,"constant_count":c,"output_map":self.map,"output_map_sha256":self.map_digest,
            "zero_mask":self.map.iter().map(|m|m[0]==0).collect::<Vec<_>>(),
            "constant_mask":self.map.iter().map(|m|m[0]==2).collect::<Vec<_>>(),
            "expressions":self.expressions.iter().map(|a|a.to_canonical_string()).collect::<Vec<_>>(),
            "dynamic_expressions":self.dynamic_expressions.iter().map(|a|a.to_canonical_string()).collect::<Vec<_>>(),
            "constant_expressions":self.constant_expressions.iter().map(|a|a.to_canonical_string()).collect::<Vec<_>>(),
            "dynamic_program":self.dynamic.artifact(),"constant_program":self.constants.as_ref().map(Program::artifact),
            "projected_instruction_lane_visits":1297408u64*(self.dynamic.inventory.instructions as u64),
            "projected_dynamic_value_visits":1297408u64*(d as u64),
            "geometry_programs":self.geometry.iter().map(Program::artifact).collect::<Vec<_>>(),
            "working_precision_bits":128,"input_precision_bits":53,"input_encoding":"nodes_weight_f64",
            "quadrature_executed":false,
            "kernel_reuse":"existing N4/h0 source tables then Symbolica -pi*d/dF; complete UV/finite contact terms",
            "domain_requirement":"fixed real EC27 positive-F unit-power h1; no reducer or scalar master calls"})
    }
}


/// Reuse e6_parameter_series.rs exact Atom stick/Symanzik construction; no
/// integration or new polynomial engine. Jacobi weights already own Jacobian.
fn geometry_program(reverse:bool)->Answer<Program> {
    let u=(0..5).map(|i|existing::variable(&format!("geometry_u{i}"))).collect::<Vec<_>>();
    let mut remaining=Atom::num(1);let mut x=Vec::with_capacity(6);
    for coordinate in &u {x.push(&remaining*coordinate);remaining*=Atom::num(1)-coordinate;}
    x.push(remaining);if reverse{x.reverse();}
    let routing:Vec<[Atom;4]>=vec![[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,1,1,0]]
        .into_iter().map(|q|q.map(Atom::num)).collect();
    let mut routing=routing;routing[4][0]=Atom::num(Rational::from((1,2)));
    let masses=[1,2,3,4,2,3].map(Atom::num);
    let mean:[Atom;4]=std::array::from_fn(|a|Atom::add_many(&(0..6).map(|i|&x[i]*&routing[i][a]).collect::<Vec<_>>()));
    let square=|q:&[Atom;4]|q[0].pow(2)-q[1].pow(2)-q[2].pow(2)-q[3].pow(2);
    let f=Atom::add_many(&(0..6).map(|i|&x[i]*(&masses[i]-square(&routing[i]))).collect::<Vec<_>>())+square(&mean);
    let values=mean.into_iter().chain([f]).collect::<Vec<_>>();
    Program::new(&values,&u)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn native_D6_full_UV_support_phase_and_tensor_factorial() {
        let p=Prepared::new().unwrap();assert_eq!(p.basis.len(),210);assert_eq!(p.expressions.len(),840);
        let symbols=existing::Symbols::new();
        let scalar=symbolica::parse!("𝑖")*Atom::var(Symbol::PI).pow(3)*Atom::num(2)/symbols.f.pow(3);
        assert!(existing::exact_zero(&(&p.expressions[3]-&scalar)));
        assert!(existing::exact_zero(&(&p.expressions[7]+&symbols.p[0]*&scalar)));
        for (alpha,slots) in p.basis.iter().zip(p.expressions.chunks_exact(4)) {
            let expected=alpha.iter().sum::<u16>()==6&&alpha.iter().all(|power|power%2==0);
            assert_eq!(!slots[0].is_zero(),expected);assert!(slots[1].is_zero()&&slots[2].is_zero());
        }
        let temporal=p.basis.iter().position(|a|*a==[6,0,0,0]).unwrap();
        let expected=-symbolica::parse!("𝑖")*Atom::var(Symbol::PI).pow(3)*Atom::num(Rational::from((15,8)));
        assert!(existing::exact_zero(&(&p.expressions[4*temporal]-expected)));
    }
}
