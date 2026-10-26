//! Source-only weighted native stream. All numeric operations use Symbolica Float.
use crate::{Answer,kernel::{Prepared,OUTPUTS}};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::io::{Read,Write};
use symbolica::{domains::float::{Complex,Float,FloatLike,Real,SingleFloat},evaluate::BatchExpressionEvaluator};

pub const PROTOCOL:&str="n6-d6-weighted-float128-v1";
pub const MAX_ROWS:usize=256;
pub const MAX_REQUEST:usize=20489;
pub const MAX_RESPONSE:usize=4_194_304;
pub const NUMBER_BYTES:usize=512;
const WINDOW_BATCHES:usize=128;
fn zero()->Float {Float::new(128)}
fn czero()->Complex<Float> {Complex::new(zero(),zero())}
fn sha(raw:&[u8])->String {format!("{:x}",Sha256::digest(raw))}
fn hex(raw:&[u8])->String {raw.iter().map(|b|format!("{b:02x}")).collect()}
fn num(v:&Float)->Answer<Value> {
    if !v.is_finite() || v.get_precision()==0 || v.get_precision()>128 {return Err("invalid native scalar/precision".into());}
    let text=v.try_to_rational().ok_or("nonfinite rational witness")?.to_string();
    if text.len()>NUMBER_BYTES {return Err("native rational exceeds fixed serialization bound".into());}
    Ok(json!([text,v.get_precision(),v.is_sign_negative()]))
}
fn complex(v:&Complex<Float>)->Answer<Value> {Ok(json!([num(&v.re)?,num(&v.im)?]))}
fn complexes(v:&[Complex<Float>])->Answer<Vec<Value>> {v.iter().map(complex).collect()}
fn numbers(v:&[Float])->Answer<Vec<Value>> {v.iter().map(num).collect()}
fn l1(v:&Complex<Float>)->Float {v.re.norm()+v.im.norm()}
fn add_complex(a:&mut Complex<Float>,b:&Complex<Float>) {a.re+=&b.re;a.im+=&b.im;}
fn add_tracked(a:&mut Float,b:&Float,minimum:&mut u32) {
    *minimum=(*minimum).min(a.get_precision()).min(b.get_precision());*a+=b;*minimum=(*minimum).min(a.get_precision());
}
fn add_complex_tracked(a:&mut Complex<Float>,b:&Complex<Float>,minimum:&mut [u32;3]) {
    add_tracked(&mut a.re,&b.re,&mut minimum[0]);add_tracked(&mut a.im,&b.im,&mut minimum[1]);
}
fn weighted(v:&Complex<Float>,w:&Float)->Complex<Float> {Complex::new(v.re.clone()*w,v.im.clone()*w)}

pub fn read_frame<R:Read>(input:&mut R)->Answer<Option<Vec<u8>>> {
    let mut prefix=[0u8;8];let mut have=0;
    while have<8 {match input.read(&mut prefix[have..]) {
        Ok(0) if have==0=>return Ok(None),Ok(0)=>return Err("truncated length".into()),Ok(n)=>have+=n,
        Err(e) if e.kind()==std::io::ErrorKind::Interrupted=>{},Err(e)=>return Err(e.to_string())}}
    let size=usize::try_from(u64::from_le_bytes(prefix)).map_err(|_|"frame length conversion")?;
    if !(9..=MAX_REQUEST).contains(&size) {return Err("request outside fixed length bound".into());}
    let mut frame=vec![0;size];input.read_exact(&mut frame).map_err(|e|e.to_string())?;Ok(Some(frame))
}
fn write_frame<W:Write>(output:&mut W,code:u8,rows:usize,cols:usize,body:&[u8])->Answer<()> {
    let length=9usize.checked_add(body.len()).ok_or("response overflow")?;
    if length>MAX_RESPONSE {return Err("response exceeds fixed limit".into());}
    let rows=u32::try_from(rows).map_err(|_|"row header overflow")?;
    let cols=u32::try_from(cols).map_err(|_|"column header overflow")?;
    output.write_all(&(length as u64).to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(&[code]).map_err(|e|e.to_string())?;
    output.write_all(&rows.to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(&cols.to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(body).map_err(|e|e.to_string())?;output.flush().map_err(|e|e.to_string())
}
fn send<W:Write>(output:&mut W,code:u8,rows:usize,cols:usize,value:&Value)->Answer<()> {
    write_frame(output,code,rows,cols,&serde_json::to_vec(value).map_err(|e|e.to_string())?)
}

struct Rule {
    index:usize,order:usize,permutation:Vec<usize>,expected_rows:usize,
    rows:usize,batches:usize,windows:usize,window_first:usize,window_rows:usize,
    weight_sum:Float,window_weight:Float,window_values:Vec<Complex<Float>>,window_abs:Vec<Float>,
    totals:Vec<Complex<Float>>,absolute:Vec<Float>,constant_totals:Vec<Complex<Float>>,constant_abs:Vec<Float>,
    min_f:Option<Float>,max_f:Option<Float>,min_p0:Option<Float>,max_p0:Option<Float>,
    input_hash:Sha256,chain:[u8;32],window_chain:[u8;32],dynamic_min:Vec<[u32;3]>,constant_min:Vec<[u32;3]>,weight_min:u32,
}
impl Rule {
    fn new(index:usize,d:usize,c:usize)->Self {
        let mut seed=PROTOCOL.as_bytes().to_vec();seed.extend_from_slice(&(index as u32).to_le_bytes());
        let chain:[u8;32]=Sha256::digest(seed).into();
        Self{index,order:if index==0{12}else{16},permutation:if index==0{(0..6).collect()}else{(0..6).rev().collect()},
            expected_rows:if index==0{248832}else{1048576},rows:0,batches:0,windows:0,window_first:1,window_rows:0,
            weight_sum:zero(),window_weight:zero(),window_values:vec![czero();d],window_abs:vec![zero();d],
            totals:vec![czero();d],absolute:vec![zero();d],constant_totals:vec![czero();c],constant_abs:vec![zero();c],
            min_f:None,max_f:None,min_p0:None,max_p0:None,input_hash:Sha256::new(),chain,window_chain:chain,dynamic_min:vec![[128;3];d],constant_min:vec![[128;3];c],weight_min:128}
    }
    fn cumulative_hash(&self)->String {format!("{:x}",self.input_hash.clone().finalize())}
    fn checkpoint(&mut self,constants:&[Complex<Float>])->Answer<Value> {
        if self.window_rows==0 {return Ok(Value::Null);}
        let mut constant_values=Vec::with_capacity(constants.len());let mut constant_abs=Vec::with_capacity(constants.len());
        for (i,value) in constants.iter().enumerate() {
            let sum=weighted(value,&self.window_weight);
            let abs=l1(value)*&self.window_weight;
            add_complex_tracked(&mut self.constant_totals[i],&sum,&mut self.constant_min[i]);add_tracked(&mut self.constant_abs[i],&abs,&mut self.constant_min[i][2]);
            constant_values.push(sum);constant_abs.push(abs);
        }
        for i in 0..self.totals.len() {add_complex_tracked(&mut self.totals[i],&self.window_values[i],&mut self.dynamic_min[i]);
            add_tracked(&mut self.absolute[i],&self.window_abs[i],&mut self.dynamic_min[i][2]);}
        add_tracked(&mut self.weight_sum,&self.window_weight,&mut self.weight_min);
        self.windows+=1;
        let record=json!({"window_index":self.windows,"first_batch":self.window_first,"last_batch":self.batches,
            "rows":self.window_rows,"cumulative_rows":self.rows,"weight_sum":num(&self.window_weight)?,
            "chain_before_first":hex(&self.window_chain),"chain_after_last":hex(&self.chain),
            "cumulative_input_sha256":self.cumulative_hash(),
            "dynamic_weighted_sums":complexes(&self.window_values)?,"dynamic_absolute_l1_sums":numbers(&self.window_abs)?,
            "constant_weighted_sums":complexes(&constant_values)?,"constant_absolute_l1_sums":numbers(&constant_abs)?,
            "dynamic_global_min_precision_bits":self.dynamic_min,"constant_global_min_precision_bits":self.constant_min,
            "weight_sum_global_min_precision_bits":self.weight_min});
        self.window_rows=0;self.window_first=self.batches+1;self.window_weight=zero();self.window_chain=self.chain;
        self.window_values.fill(czero());self.window_abs.fill(zero());Ok(record)
    }
}

pub struct Worker<'a> {
    prepared:&'a Prepared,evaluator:BatchExpressionEvaluator<Complex<Float>>,geometry:[BatchExpressionEvaluator<Complex<Float>>;2],constants:Vec<Complex<Float>>,
    rule:Option<Rule>,completed:usize,
}
impl<'a> Worker<'a> {
    pub fn new(prepared:&'a Prepared)->Answer<Self> {
        let evaluator=prepared.dynamic.instantiate()?;
        let constants=if let Some(program)=&prepared.constants {
            let mut program=program.instantiate()?;
            let point=vec![czero(),czero(),czero(),czero(),Complex::new(Float::with_val(128,1),zero())];
            let mut values=vec![czero();prepared.constant_expressions.len()];
            program.evaluate_batch(&point,&mut values,1).map_err(|e|e.to_string())?;values
        }else{Vec::new()};
        if constants.iter().any(|v|!v.is_finite()) {return Err("nonfinite mapped constant".into());}
        let geometry=[prepared.geometry[0].instantiate()?,prepared.geometry[1].instantiate()?];
        Ok(Self{prepared,evaluator,geometry,constants,rule:None,completed:0})
    }
    pub fn ready(&self)->Answer<Value> {
        Ok(json!({"protocol":PROTOCOL,"topology":6,"requested_topology":6,"dimension_shift_h":1,
            "parameters":5,"input_columns":10,"input_encoding":"nodes_weight_f64","geometry_arithmetic":"Symbolica_Float128","moments":210,"native_slots":4,"outputs":840,"basis":self.prepared.basis,
            "poles":["uv1","ir1","ir2","finite"],"output_order":"moment_major_native_uv1_ir1_ir2_finite",
            "dynamic_count":self.prepared.dynamic_expressions.len(),"constant_count":self.constants.len(),
            "constant_values":complexes(&self.constants)?,"output_map":self.prepared.map,"output_map_sha256":self.prepared.map_digest,
            "zero_mask":self.prepared.map.iter().map(|m|m[0]==0).collect::<Vec<_>>(),
            "constant_mask":self.prepared.map.iter().map(|m|m[0]==2).collect::<Vec<_>>(),
            "native_normalization_included":true,"ir_poles_proven_zero":true,"uv_poles_proven_zero":false,"nonzero_degree6_UV_components":20,"working_precision_bits":128,"input_precision_bits":53,
            "parameters_are_real":true,"positive_f_lower_bound":"15/16","positive_f_upper_bound":"19/4",
            "p0_lower_bound":"0","p0_upper_bound":"1/2","window_batches":128,
            "absolute_sum_norm":"sum(abs(rounded(weight*real))+abs(rounded(weight*imag)))",
            "absolute_sum_semantics":"L1 estimate of individually rounded weighted components; not a rigorous integral bound",
            "max_number_bytes":NUMBER_BYTES,
            "max_rows":MAX_ROWS,"max_request_bytes":MAX_REQUEST,"max_response_bytes":MAX_RESPONSE,"frame_prefix_bytes":8,
            "byte_bounds_exclude_length_prefix":true,"instruction_major_symbolica_batch":true,"no_jit":true,
            "dynamic_program":self.prepared.dynamic.artifact(),"constant_program":self.prepared.constants.as_ref().map(crate::program::Program::artifact)}))
    }
    fn start(&mut self,body:&[u8])->Answer<Value> {
        if self.rule.is_some() || self.completed>=2 {return Err("rule state does not admit START".into());}
        let request:Value=serde_json::from_slice(body).map_err(|e|e.to_string())?;
        let rule=Rule::new(self.completed,self.prepared.dynamic_expressions.len(),self.constants.len());
        if request!=json!({"configuration":rule.index,"order":rule.order,"permutation":rule.permutation}) {
            return Err("START is not the exact next fixed rule".into());}
        let response=json!({"status":"rule_started","configuration":rule.index,"order":rule.order,"permutation":rule.permutation,
            "rows":0,"batches":0,"windows":0,"batch_chain_sha256":hex(&rule.chain),"cumulative_input_sha256":rule.cumulative_hash()});
        self.rule=Some(rule);Ok(response)
    }
    fn evaluate(&mut self,payload:&[u8],rows:usize)->Answer<Value> {
        let rule=self.rule.as_mut().ok_or("EVAL without START")?;
        if rows==0 || rows>MAX_ROWS || rows.checked_mul(80).and_then(|n|n.checked_add(9))!=Some(payload.len())
            || rule.rows.checked_add(rows).ok_or("row overflow")?>rule.expected_rows {return Err("EVAL dimensions or rule row bound differs".into());}
        let mut parameters=Vec::with_capacity(rows*5);let mut weights=Vec::with_capacity(rows);
        for row in payload[9..].chunks_exact(80) {
            let values=row.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().expect("exact8bytechunk"))).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) || values[..5].iter().any(|v|*v<=0.0||*v>=1.0)
                || values[5..].iter().any(|w|*w<=0.0||*w>1.0) {return Err("nonfinite or inadmissible sticknode/axisweight".into());}
            parameters.extend(values[..5].iter().map(|v|Complex::new(Float::with_val(128,*v),zero())));
            let mut weight=Float::with_val(128,1);
            for value in &values[5..] {weight*=Float::with_val(128,*value);}
            weights.push(weight);
        }
        let mut physical=vec![czero();rows*5];
        self.geometry[rule.index].evaluate_batch(&parameters,&mut physical,rows).map_err(|e|e.to_string())?;
        let lower=Float::with_val(128,15)/Float::with_val(128,16);let upper=Float::with_val(128,19)/Float::with_val(128,4);
        let half=Float::with_val(128,1)/Float::with_val(128,2);
        for p in physical.chunks_exact(5) {
            if p.iter().any(|v|!v.is_finite()||!v.im.is_zero())||p[4].re<lower||p[4].re>upper||p[0].re<zero()||p[0].re>half {
                return Err("Float128 geometry violates fixed positive-F/P0 domain".into());}
            if rule.min_f.as_ref().is_none_or(|v|p[4].re<*v){rule.min_f=Some(p[4].re.clone());}
            if rule.max_f.as_ref().is_none_or(|v|p[4].re>*v){rule.max_f=Some(p[4].re.clone());}
            if rule.min_p0.as_ref().is_none_or(|v|p[0].re<*v){rule.min_p0=Some(p[0].re.clone());}
            if rule.max_p0.as_ref().is_none_or(|v|p[0].re>*v){rule.max_p0=Some(p[0].re.clone());}
        }
        let d=self.prepared.dynamic_expressions.len();let mut values=vec![czero();rows*d];
        self.evaluator.evaluate_batch(&physical,&mut values,rows).map_err(|e|e.to_string())?;
        if values.iter().any(|v|!v.is_finite()) {return Err("nonfinite dynamic point output".into());}
        for (row,weight) in weights.iter().enumerate() {
            rule.window_weight+=weight;
            for i in 0..d {
                let value=weighted(&values[row*d+i],weight);
                add_complex(&mut rule.window_values[i],&value);rule.window_abs[i]+=l1(&value);
            }
        }
        rule.rows+=rows;rule.window_rows+=rows;rule.batches+=1;
        let input=Sha256::digest(payload);rule.input_hash.update(payload);
        let mut chain=Sha256::new();chain.update(rule.chain);chain.update((payload.len() as u64).to_le_bytes());
        chain.update((rows as u32).to_le_bytes());chain.update(input);rule.chain=chain.finalize().into();
        let checkpoint=if rule.batches%WINDOW_BATCHES==0 {rule.checkpoint(&self.constants)?}else{Value::Null};
        Ok(json!({"status":"batch_accumulated","configuration":rule.index,"batch_index":rule.batches,"rows":rows,
            "cumulative_rows":rule.rows,"cumulative_batches":rule.batches,"input_payload_bytes":payload.len(),
            "input_sha256":hex(&input),"cumulative_input_sha256":rule.cumulative_hash(),"batch_chain_sha256":hex(&rule.chain),
            "checkpoint":checkpoint}))
    }
    fn finish(&mut self)->Answer<Value> {
        let rule=self.rule.as_mut().ok_or("FINAL without START")?;
        let expected_batches=if rule.index==0{972}else{4096};
        if rule.rows!=rule.expected_rows || rule.batches!=expected_batches {return Err("FINAL before exact fixed row/batch inventory".into());}
        let checkpoint=rule.checkpoint(&self.constants)?;
        if rule.windows!=if rule.index==0{8}else{32} {return Err("window inventory differs".into());}
        let mut values=Vec::with_capacity(OUTPUTS);let mut absolute=Vec::with_capacity(OUTPUTS);let mut minima=Vec::with_capacity(OUTPUTS);
        for &[kind,index] in &self.prepared.map {
            let (value,abs)=match kind {0=>(czero(),zero()),1=>(rule.totals[index].clone(),rule.absolute[index].clone()),
                2=>(rule.constant_totals[index].clone(),rule.constant_abs[index].clone()),_=>return Err("invalid prepared map kind".into())};
            if !abs.is_finite() || abs<zero() {return Err("invalid accumulated L1 sum".into());}
            values.push(value);absolute.push(abs);minima.push(match kind{0=>[128;3],1=>rule.dynamic_min[index],2=>rule.constant_min[index],_=>unreachable!()});
        }
        let result=json!({"status":"rule_complete","configuration":rule.index,"order":rule.order,"permutation":rule.permutation,
            "rows":rule.rows,"batches":rule.batches,"windows":rule.windows,"weight_sum":num(&rule.weight_sum)?,
            "cumulative_input_sha256":rule.cumulative_hash(),"batch_chain_sha256":hex(&rule.chain),
            "output_map_sha256":self.prepared.map_digest,"weighted_sums":complexes(&values)?,
            "absolute_l1_sums":numbers(&absolute)?,"checkpoint":checkpoint,
            "accumulation_min_precision_bits":minima,"weight_sum_global_min_precision_bits":rule.weight_min,
            "geometry_extrema":{"minimum_F":num(rule.min_f.as_ref().ok_or("missing minimum F")?)?,
                "maximum_F":num(rule.max_f.as_ref().ok_or("missing maximum F")?)?,
                "minimum_P0":num(rule.min_p0.as_ref().ok_or("missing minimum P0")?)?,
                "maximum_P0":num(rule.max_p0.as_ref().ok_or("missing maximum P0")?)?}});
        self.completed+=1;self.rule=None;Ok(result)
    }
    pub fn serve<R:Read,W:Write>(&mut self,input:&mut R,output:&mut W)->Answer<()> {
        send(output,2,0,0,&self.ready()?)?;
        loop {
            let frame=match read_frame(input) {
                Ok(Some(v))=>v,Ok(None) if self.rule.is_none()&&self.completed==2=>return Ok(()),
                Ok(None)=>return Err("EOF before complete fixed cohort".into()),Err(e)=>{write_frame(output,1,0,0,e.as_bytes())?;return Err(e);}};
            let code=frame[0];let rows=u32::from_le_bytes(frame[1..5].try_into().map_err(|_|"rowheader")?) as usize;
            let cols=u32::from_le_bytes(frame[5..9].try_into().map_err(|_|"colheader")?) as usize;
            let answer=(||->Answer<Option<(u8,usize,usize,Value)>> {match code {
                10 if rows==0&&cols==0=>Ok(Some((10,0,0,self.start(&frame[9..])?))),
                1 if cols==10=>Ok(Some((0,rows,0,self.evaluate(&frame,rows)?))),
                4 if rows==0&&cols==0&&frame.len()==9=>Ok(Some((4,0,OUTPUTS,self.finish()?))),
                2 if rows==0&&cols==0&&frame.len()==9&&self.rule.is_none()&&self.completed==2=>Ok(None),
                _=>Err("invalid opcode/shape/state".into())}})();
            match answer {
                Ok(Some((code,rows,cols,value)))=>send(output,code,rows,cols,&value)?,
                Ok(None)=>{write_frame(output,3,0,0,&[])?;return Ok(());},
                Err(e)=>{write_frame(output,1,0,0,e.as_bytes())?;return Err(e);}
            }
        }
    }
}


#[cfg(test)] mod tests {
    use super::*;
    use symbolica::domains::rational::Rational;
    #[test] fn wire_lift_is_exact_binary64_at128_not53() {
        let v=Float::with_val(128,f64::from_bits(0x3ff0000000000001));
        assert_eq!(v.get_precision(),128);
        assert_eq!(v.to_rational(),Rational::from((4503599627370497_i64,4503599627370496_i64)));
        let negative_zero=Float::with_val(128,-0.0);assert_eq!(negative_zero.get_precision(),128);
        assert!(negative_zero.is_sign_negative());assert_eq!(negative_zero.to_rational(),Rational::from(0));
    }
}
