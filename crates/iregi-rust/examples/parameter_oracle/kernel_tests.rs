//! Exact independent source checks. These tests import no reducer or scalar master.
use super::*;
use std::sync::OnceLock;
fn equal(a:&Atom,b:&Atom){assert!(exact_zero(&(a-b)),"{} != {}",a,b);}
fn prepared(n:usize)->&'static Prepared {
    static N2:OnceLock<Prepared>=OnceLock::new();static N3:OnceLock<Prepared>=OnceLock::new();static N4:OnceLock<Prepared>=OnceLock::new();
    match n {2=>&N2,3=>&N3,4=>&N4,_=>panic!("test topology")}.get_or_init(||Prepared::new(n).unwrap())
}
fn at(p:&Prepared,alpha:[u16;4],slot:usize)->&Atom {
    &p.expressions[4*p.basis.iter().position(|&b|b==alpha).unwrap()+slot]
}
fn derivative_at(expression:&Atom,alpha:[u16;4],s:&Symbols)->Atom {
    let mut value=expression.clone();
    for (axis,&power) in alpha.iter().enumerate(){for _ in 0..power{value=value.derivative(s.t_symbols[axis]);}}
    substitute(&value,&s.t.iter().cloned().map(|t|(t,Atom::num(0))).collect::<Vec<_>>()).expand().together()
}
#[test]
fn builtin_gamma_laurent_regularizer_handles_every_needed_nonpositive_integer(){
    let s=Symbols::new();let gamma_e=Atom::from(symbolica::transcendental::euler_gamma());
    let residues=[Atom::num(1),Atom::num(-1),Atom::num(1)/Atom::num(2),-Atom::num(1)/Atom::num(6)];
    let finite=[-&gamma_e,&gamma_e-Atom::num(1),Atom::num(3)/Atom::num(4)-&gamma_e/Atom::num(2),&gamma_e/Atom::num(6)-Atom::num(11)/Atom::num(36)];
    for k in 0..4 {
        let value=symbolica::transcendental::gamma().call((&s.epsilon-Atom::num(k as i64),));
        let series=value.series(s.epsilon_symbol,0,0).unwrap();
        equal(&series.coefficient((-2).into()).unwrap(),&Atom::new());
        equal(&series.coefficient((-1).into()).unwrap(),&residues[k]);
        equal(&series.coefficient(0.into()).unwrap(),&finite[k]);
        assert!(series.coefficient(1.into()).is_none(),"unknown higher coefficient must stay unknown");
    }
}
#[test]
fn scalar_laurent_integrands_fix_native_phase_scale_and_euler_constant(){
    let phase=native_phase();let gamma_e=Atom::from(symbolica::transcendental::euler_gamma());
    for n in [2,3,4] {
        let p=prepared(n);let f=&p.symbols.f;
        let expected=match n {
            2=>[phase.clone(),&phase*(physical_log(&p.symbols)-&gamma_e)],
            3=>[Atom::new(),-&phase/f],
            4=>[Atom::new(),&phase/f.pow(2)],_=>unreachable!()};
        equal(at(p,[0;4],0),&expected[0]);equal(at(p,[0;4],3),&expected[1]);
    }
}
#[test]
fn low_rank_uv_metric_residues_fix_wick_sign_and_tensor_factorials(){
    let phase=native_phase();
    for n in [2,3] {
        let p=prepared(n);let origin=p.symbols.p.iter().cloned().map(|p|(p,Atom::num(0))).collect::<Vec<_>>();
        let radial=if n==2 {&phase*&p.symbols.f/Atom::num(2)}else{&phase/Atom::num(2)};
        equal(&substitute(at(p,[2,0,0,0],0),&origin),&radial);
        equal(&substitute(at(p,[0,2,0,0],0),&origin),&(-&radial));
        equal(&substitute(at(p,[1,1,0,0],0),&origin),&Atom::new());
    }
    let p=prepared(4);let origin=p.symbols.p.iter().cloned().map(|p|(p,Atom::num(0))).collect::<Vec<_>>();
    for (alpha,num,den) in [([4,0,0,0],3,4),([0,4,0,0],3,4),([2,2,0,0],-1,4)] {
        equal(&substitute(at(p,alpha,0),&origin),&(&phase*Atom::num(num)/Atom::num(den)));
    }
}
#[test]
fn every_330_source_coefficient_matches_repeated_symbolica_derivatives(){
    for n in [2,3,4] {
        let p=prepared(n);let source=laurent_kernels(n,&p.symbols,false).unwrap();
        for &alpha in &p.basis {
            equal(at(p,alpha,0),&derivative_at(&source[0],alpha,&p.symbols));
            equal(at(p,alpha,3),&derivative_at(&source[1],alpha,&p.symbols));
        }
    }
}
#[test]
fn source_then_epsilon_expansion_commutes_with_epsilon_then_source_for_full_n4(){
    let p=prepared(4);let s=&p.symbols;
    let source_first=generating_kernel(4,s,true).unwrap().series(s.lambda_symbol,0,7).unwrap().to_atom();
    let epsilon=source_first.series(s.epsilon_symbol,0,0).unwrap();
    for (slot,power) in [(0,-1),(3,0)] {
        let expression=epsilon.coefficient(power.into()).unwrap();
        let physical=substitute(&expression,&[(s.ell.clone(),physical_log(s))]);
        let (coefficients,_)=source_coefficients(&physical,s).unwrap();
        for (&alpha,value) in p.basis.iter().zip(coefficients){equal(at(p,alpha,slot),&value);}
    }
}
#[test]
fn full_native_layout_proves_ir_zeros_and_retains_uv_where_permitted(){
    for n in [2,3,4] {
        let p=prepared(n);assert_eq!(p.expressions.len(),1320);assert_eq!(p.basis.len(),330);
        assert!(p.expressions.chunks_exact(4).all(|slots|slots[1].is_zero()&&slots[2].is_zero()));
        let mut nonzero_uv=0;
        for (alpha,slots) in p.basis.iter().zip(p.expressions.chunks_exact(4)) {
            if i32::from(alpha.iter().sum::<u16>())+4-2*(n as i32)<0 {assert!(exact_zero(&slots[0]));}
            else if !exact_zero(&slots[0]){nonzero_uv+=1;}
            assert!(!slots[0].contains(&p.symbols.epsilon)&&!slots[3].contains(&p.symbols.epsilon));
        }
        assert!(nonzero_uv>0,"UV-bearing low-point family cannot silently become all zero");
    }
}
#[test]
fn source_basis_order_and_raw_multiplicities_are_unique(){
    let b=basis();assert_eq!(b.len(),330);assert_eq!(b[0],[0;4]);assert_eq!(b[35],[4,0,0,0]);
    assert_eq!(b[126],[6,0,0,0]);assert_eq!(b[210],[7,0,0,0]);assert_eq!(b[273],[1,3,2,1]);
    assert_eq!(b[276],[1,2,4,0]);assert_eq!(b[329],[0,0,0,7]);
    let mut unique=b.clone();unique.sort();unique.dedup();assert_eq!(unique.len(),330);
}
#[test]
fn all_moments_obey_loop_reversal_and_spatial_axis_permutation(){
    let p=prepared(4);let s=&p.symbols;
    let reverse=s.p.iter().map(|x|(x.clone(),-x)).collect::<Vec<_>>();
    let swap=[(s.p[1].clone(),s.p[2].clone()),(s.p[2].clone(),s.p[1].clone())];
    for &alpha in &p.basis {for slot in [0,3] {
        let value=at(p,alpha,slot);let sign=Atom::num(if alpha.iter().sum::<u16>()%2==0 {1}else{-1});
        equal(&substitute(value,&reverse),&(value*sign));
        equal(&substitute(value,&swap),at(p,[alpha[0],alpha[2],alpha[1],alpha[3]],slot));
    }}
}
fn minkowski_square(p:&[Atom;4])->Atom {p[0].pow(2)-p[1].pow(2)-p[2].pow(2)-p[3].pow(2)}
#[test]
fn complex_fixture_pair_and_completed_square_F_agree_exactly_in_symbolica(){
    for n in [2,3,4] {
        let points:[[i64;4];4]=[[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
        let point=points[..n].iter().map(|row|row.map(Atom::num)).collect::<Vec<_>>();
        let x=(0..n).map(|i|variable(&format!("x{i}"))).collect::<Vec<_>>();
        let masses=(1..=n).map(|i|Atom::num(i as i64)*(Atom::num(1)-symbolica::parse!("𝑖")/Atom::num(8))).collect::<Vec<_>>();
        let sum=Atom::add_many(&x);let mean:[Atom;4]=std::array::from_fn(|a|Atom::add_many(&(0..n).map(|i|&x[i]*&point[i][a]).collect::<Vec<_>>()));
        let direct=&sum*Atom::add_many(&(0..n).map(|i|&x[i]*(&masses[i]-minkowski_square(&point[i]))).collect::<Vec<_>>())+minkowski_square(&mean);
        let mut distances=Vec::new();for i in 0..n{for j in i+1..n{
            let delta=std::array::from_fn(|axis|&point[i][axis]-&point[j][axis]);distances.push(&x[i]*&x[j]*minkowski_square(&delta));
        }}
        let pair=&sum*Atom::add_many(&(0..n).map(|i|&x[i]*&masses[i]).collect::<Vec<_>>())-Atom::add_many(&distances);equal(&direct,&pair);
        let last=Atom::num(1)-Atom::add_many(&x[..n-1]);
        let ordinary=Atom::add_many(&(0..n).map(|i|&x[i]*(&masses[i]-minkowski_square(&point[i]))).collect::<Vec<_>>())+minkowski_square(&mean);
        equal(&substitute(&direct,&[(x[n-1].clone(),last.clone())]),&substitute(&ordinary,&[(x[n-1].clone(),last)]));
    }
}
#[test]
fn unsupported_topologies_fail_before_evaluator_preparation(){
    for n in [0,1,5,6,7,8]{assert!(validate_domain(n).is_err());assert!(Prepared::new(n).is_err());}
}
