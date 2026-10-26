//! SOURCE ONLY: independent N5/D6 parameter kernels and exact gap Taylor map.
//! No reducer/master imports, no integral evaluation, and no custom algebra.
#[path="kernel_prefix.rs"] mod existing;
use crate::{Answer,program::Program};
use serde_json::{Value,json};
use std::collections::HashMap;
use symbolica::{atom::{Atom,AtomCore,Symbol},domains::rational::Rational};
pub const MOMENTS:usize=210;
pub const NATIVE:usize=4;

/// Exact output indexing only. Symbolica owns expression normalization.
pub struct Table {
    pub expressions:Vec<Atom>,pub map:Vec<[usize;2]>,
    pub dynamic_expressions:Vec<Atom>,pub constant_expressions:Vec<Atom>,
    pub dynamic:Option<Program>,pub constants:Option<Program>,
}
impl Table {
    fn new(expressions:Vec<Atom>,parameters:&[Atom])->Answer<Self> {
        let mut map=Vec::with_capacity(expressions.len());
        let mut dynamic_expressions=Vec::new();let mut constant_expressions=Vec::new();
        let mut dynamic_index=HashMap::<Atom,usize>::new();let mut constant_index=HashMap::<Atom,usize>::new();
        for expression in &expressions {
            if expression.is_zero() {map.push([0,0]);continue;}
            let dynamic=parameters.iter().any(|parameter|expression.contains(parameter));
            let (table,index,kind)=if dynamic {(&mut dynamic_expressions,&mut dynamic_index,1)}
                else {(&mut constant_expressions,&mut constant_index,2)};
            let slot=*index.entry(expression.clone()).or_insert_with(||{let slot=table.len();table.push(expression.clone());slot});
            map.push([kind,slot]);
        }
        let dynamic=if dynamic_expressions.is_empty(){None}else{Some(Program::new(&dynamic_expressions,parameters)?)};
        let constants=if constant_expressions.is_empty(){None}else{Some(Program::new(&constant_expressions,parameters)?)};
        Ok(Self{expressions,map,dynamic_expressions,constant_expressions,dynamic,constants})
    }
    fn artifact(&self)->Value {json!({"outputs":self.expressions.len(),"map":self.map,
        "dynamic_count":self.dynamic_expressions.len(),"constant_count":self.constant_expressions.len(),
        "dynamic_program":self.dynamic.as_ref().map(Program::artifact),
        "constant_program":self.constants.as_ref().map(Program::artifact)})}
}
pub struct Prepared {
    pub basis:Vec<[u16;4]>,pub unit:Table,pub first_second_f_derivatives:Table,
    pub geometry:[Program;2],pub gap_taylor_map:Program,
}
impl Prepared {
    pub fn new()->Answer<Self> {
        let s=existing::Symbols::new();
        let f_symbol=symbolica::symbol!("low_n_parameter_oracle::F");
        if Atom::var(f_symbol)!=s.f{return Err("F namespace differs from qualified source".into());}
        let [uv,finite]=existing::laurent_kernels(4,&s,true)?;
        let(uv,_)=existing::source_coefficients(&uv,&s)?;
        let(finite,_)=existing::source_coefficients(&finite,&s)?;
        if uv.len()!=330 || finite.len()!=330{return Err("incomplete original N4 source tables".into());}
        let basis=existing::basis().into_iter().take(MOMENTS).collect::<Vec<_>>();
        if basis.len()!=MOMENTS || basis.iter().any(|alpha|alpha.iter().sum::<u16>()>6){return Err("rank6 basis differs".into());}
        let factor=-Atom::var(Symbol::PI);
        let mut unit=Vec::with_capacity(MOMENTS*NATIVE);
        let mut derivatives=Vec::with_capacity(MOMENTS*NATIVE*2);
        for index in 0..MOMENTS {
            let native=[(&factor*&uv[index]).expand().together(),Atom::new(),Atom::new(),
                (&factor*&finite[index]).expand().together()];
            for slot in &native {
                let first=slot.derivative(f_symbol).expand().together();
                let second=(&first.derivative(f_symbol)/Atom::num(2)).expand().together();
                derivatives.extend([first,second]);
            }
            unit.extend(native);
        }
        let unit=Table::new(unit,&s.parameters())?;
        let first_second_f_derivatives=Table::new(derivatives,&s.parameters())?;
        let geometry=[geometry(false)?,geometry(true)?];
        let gap_taylor_map=gap_program()?;
        Ok(Self{basis,unit,first_second_f_derivatives,geometry,gap_taylor_map})
    }
    pub fn artifact(&self)->Value {json!({"status":"SOURCE_KERNELS_prepared_not_integrated",
        "independent_numeric_agreement_qualified":false,"moments":MOMENTS,"native_slots":NATIVE,
        "basis":self.basis,"unit_layout":"moment,UV1/IR1/IR2/finite",
        "f_derivative_layout":"moment,native_slot,first/second_Taylor_F_derivative",
        "mass_weight_factors":"geometry x0 multiplies first_F; geometry x0_squared multiplies second_Taylor_F",
        "native_child_normalization":"N5D6 = -pi * existing N4D4 Gaussian source; all UV and finite contact terms retained",
        "native_parent_normalization":"unchanged dimensional measure; exact duplicate-origin denominator identity, no extra pi or Gamma factor",
        "parent_columns":"unit,d/dmass0_squared,(d/dmass0_squared)^2/2",
        "physical_scale_mu":"3","physical_log":"log(9)-log(pi)-log(F)",
        "unit":self.unit.artifact(),"f_derivatives":self.first_second_f_derivatives.artifact(),
        "geometry_programs":self.geometry.iter().map(Program::artifact).collect::<Vec<_>>(),
        "gap_taylor_program":self.gap_taylor_map.artifact(),
        "integration_rules":"pending separate source admission: N5 order12 identity and16 reverse,20736+65536 common simplex points",
        "quadrature_executed":false,"runtime_cost_measured":false,"no_reducer_or_master_imports":true})}
}

/// Existing Symbolica stick/Symanzik construction; no Jacobian duplicated.
/// Input5 is a deliberately unused zero padding slot for unchanged Program API.
/// Output order P0,Px,Py,Pz,F_mass10,F_mass26,x0,x0_squared.
fn geometry(reverse:bool)->Answer<Program> {
    let u=(0..5).map(|i|existing::variable(&format!("sample_anchor_u{i}"))).collect::<Vec<_>>();
    let mut remaining=Atom::num(1);let mut x=Vec::with_capacity(5);
    for coordinate in &u[..4] {x.push(&remaining*coordinate);remaining*=Atom::num(1)-coordinate;}
    x.push(remaining);if reverse{x.reverse();}
    let routing:Vec<[Atom;4]>=vec![[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        .into_iter().map(|q|q.map(Atom::num)).collect();
    let mean:[Atom;4]=core::array::from_fn(|axis|Atom::add_many(&(0..5).map(|i|&x[i]*&routing[i][axis]).collect::<Vec<_>>()));
    let square=|q:&[Atom;4]|q[0].pow(2)-q[1].pow(2)-q[2].pow(2)-q[3].pow(2);
    let symanzik=|mass0:i64| {
        let masses=[mass0,12,15,18,22].map(Atom::num);
        Atom::add_many(&(0..5).map(|i|&x[i]*(&masses[i]-square(&routing[i]))).collect::<Vec<_>>())+square(&mean)
    };
    let f10=symanzik(10);let f26=symanzik(26);
    let values=mean.into_iter().chain([f10,f26,x[0].clone(),x[0].pow(2)]).collect::<Vec<_>>();
    Program::new(&values,&u)
}
fn gap_expressions()->(Vec<Atom>,Vec<Atom>) {
    let [a0,a1,a2,b0,pad]=["A0","A1","A2","B0","unused"]
        .map(|name|existing::variable(&format!("sample_gap_{name}")));
    let a_symbol=symbolica::symbol!("sample_gap_mass_shift");let a=Atom::var(a_symbol);
    let rational=(&b0-(&a0+&a*&a1+a.pow(2)*&a2))/(Atom::num(16)-&a);
    let first=rational.derivative(a_symbol);
    let second=first.derivative(a_symbol)/Atom::num(2);
    let values=[rational,first,second].map(|expression|
        existing::substitute(&expression,&[(a.clone(),Atom::num(0))]).expand().together()).to_vec();
    (vec![a0,a1,a2,b0,pad],values)
}
fn gap_program()->Answer<Program> {
    let(parameters,values)=gap_expressions();Program::new(&values,&parameters)
}

#[cfg(test)]mod source_proofs {
    use super::*;
    #[test]fn duplicate_origin_partial_fraction_and_mass_jet_identity() {
        let q=existing::variable("sample_pf_q_squared");
        let a=existing::variable("sample_pf_mass_squared_shift");
        let d0=&q-Atom::num(10)-&a;let d5=&q-Atom::num(26);
        let expected=(Atom::num(1)/&d5-Atom::num(1)/&d0)/(Atom::num(16)-&a);
        assert!(existing::exact_zero(&(Atom::num(1)/(&d0*&d5)-expected)));
        let(params,values)=gap_expressions();
        let expected=[(&params[3]-&params[0])/Atom::num(16),
            (&params[3]-&params[0])/Atom::num(256)-&params[1]/Atom::num(16),
            (&params[3]-&params[0])/Atom::num(4096)-&params[1]/Atom::num(256)-&params[2]/Atom::num(16)];
        for(actual,expected)in values.iter().zip(expected){assert!(existing::exact_zero(&(actual-expected)));}
    }
    #[test]fn n5_D6_scalar_time_and_rank4_UV_contact_are_retained() {
        let p=Prepared::new().unwrap();let s=existing::Symbols::new();
        let scalar=-symbolica::parse!("𝑖")*Atom::var(Symbol::PI).pow(3)/s.f.pow(2);
        assert!(existing::exact_zero(&(&p.unit.expressions[3]-&scalar)));
        assert!(existing::exact_zero(&(&p.unit.expressions[7]+&s.p[0]*&scalar)));
        let rank4=p.basis.iter().position(|alpha|*alpha==[4,0,0,0]).unwrap();
        let contact=-symbolica::parse!("𝑖")*Atom::var(Symbol::PI).pow(3)*Atom::num(Rational::from((3,4)));
        assert!(existing::exact_zero(&(&p.unit.expressions[rank4*NATIVE]-contact)));
        assert!(!p.unit.expressions[rank4*NATIVE].is_zero());
        for slots in p.unit.expressions.chunks_exact(4){assert!(slots[1].is_zero()&&slots[2].is_zero());}
    }
}
