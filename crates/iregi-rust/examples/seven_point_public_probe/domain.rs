//! Necessary domain audit only; no coefficient specialization or pruning.
//! Symbolica supplies exact rational arithmetic, rank, determinants and inverses.
use serde_json::{Value, json};
use std::error::Error;
use symbolica::{
    domains::rational::{Q, Rational},
    tensors::matrix::Matrix,
};

fn dot(a: &[Rational; 4], b: &[Rational; 4]) -> Rational {
    &a[0] * &b[0] - &a[1] * &b[1] - &a[2] * &b[2] - &a[3] * &b[3]
}

fn matrix_record(rows: Vec<Vec<Rational>>) -> Result<Value, Box<dyn Error>> {
    let matrix = Matrix::from_nested_vec(rows.clone(), Q)?;
    let determinant = matrix.det()?;
    let regular = determinant != Rational::from(0);
    let condition = if regular {
        let inverse = matrix.inv()?;
        let norm = rows
            .iter()
            .map(|r| r.iter().fold(Rational::from(0), |s, v| s + v.abs()))
            .max()
            .ok_or("empty matrix")?;
        let inverse_norm = (0..inverse.nrows())
            .map(|i| {
                (0..inverse.ncols()).fold(Rational::from(0), |s, j| {
                    s + inverse[(i as u32, j as u32)].abs()
                })
            })
            .max()
            .ok_or("empty inverse")?;
        Some((norm * inverse_norm).to_string())
    } else {
        None
    };
    Ok(
        json!({"rows": rows.iter().map(|r| r.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "determinant": determinant.to_string(), "regular": regular,
        "condition_infinity_exact": condition}),
    )
}

pub fn audit(fixture: &Value) -> Result<Value, Box<dyn Error>> {
    let sample = &fixture["samples"][0];
    let propagators = sample["propagators"]
        .as_array()
        .ok_or("missing propagators")?;
    if propagators.len() != 7 {
        return Err("exact domain requires seven lines".into());
    }
    let mut shifts = Vec::new();
    let mut masses = Vec::new();
    for line in propagators {
        let p = line["shift"].as_array().ok_or("missing shift")?;
        let parsed = p.iter().map(super::number).collect::<Result<Vec<_>, _>>()?;
        shifts.push(<[Rational; 4]>::try_from(parsed).map_err(|_| "shift arity")?);
        if super::number(&line["mass_squared"][1])? != Rational::from(0) {
            return Err("positive-F slice requires real masses".into());
        }
        masses.push(super::number(&line["mass_squared"][0])?);
    }
    let relative_rows = |active: &[usize]| -> Vec<Vec<Rational>> {
        (0..4)
            .map(|mu| {
                active[1..]
                    .iter()
                    .map(|&i| &shifts[i][mu] - &shifts[active[0]][mu])
                    .collect()
            })
            .collect()
    };
    let f_row = |active: &[usize]| -> Vec<Rational> {
        active[1..]
            .iter()
            .map(|&i| {
                let delta = std::array::from_fn(|mu| &shifts[i][mu] - &shifts[active[0]][mu]);
                dot(&delta, &delta) - &masses[i] + &masses[active[0]]
            })
            .collect()
    };
    let mut six = Vec::new();
    let mut six_regular = true;
    for omitted in 0..7 {
        let active: Vec<_> = (0..7).filter(|&i| i != omitted).collect();
        let mut rows = relative_rows(&active);
        rows.push(f_row(&active));
        let mut entry = matrix_record(rows)?;
        six_regular &= entry["regular"] == true;
        entry["omitted"] = json!(omitted);
        entry["active"] = json!(active);
        six.push(entry);
    }
    let mut five = Vec::new();
    let mut five_supported = true;
    for omitted_a in 0..7 {
        for omitted_b in omitted_a + 1..7 {
            let active: Vec<_> = (0..7)
                .filter(|&i| i != omitted_a && i != omitted_b)
                .collect();
            let rows = relative_rows(&active);
            let coordinate_rank = Matrix::from_nested_vec(rows.clone(), Q)?.rank();
            let coordinates = matrix_record(rows.clone())?;
            let mut extended = rows;
            extended.push(f_row(&active));
            let mut constant_charts = Vec::new();
            for dropped in 0..4 {
                let mut entry = matrix_record(
                    extended
                        .iter()
                        .enumerate()
                        .filter_map(|(i, r)| (i != dropped).then_some(r.clone()))
                        .collect(),
                )?;
                entry["dropped_coordinate_row"] = json!(dropped);
                constant_charts.push(entry);
            }
            let supported = coordinates["regular"] == true
                || (coordinate_rank == 3 && constant_charts.iter().any(|c| c["regular"] == true));
            five_supported &= supported;
            five.push(json!({"omitted":[omitted_a,omitted_b],"active":active,
                "coordinate_rank":coordinate_rank,"coordinates":coordinates,
                "constant_charts":constant_charts,"necessary_domain_supported":supported}));
        }
    }
    let minimum_mass = masses.iter().min().ok_or("missing masses")?;
    let min_time = shifts.iter().map(|p| &p[0]).min().ok_or("missing time")?;
    let max_time = shifts.iter().map(|p| &p[0]).max().ok_or("missing time")?;
    let range = max_time - min_time;
    // F = sum(x m²) + Var(spatial p) - Var(time p), and Var(t)<=range²/4.
    let lower = minimum_mass - &((&range * &range) / Rational::from(4));
    let mut q_squared = masses
        .iter()
        .map(Rational::abs)
        .max()
        .ok_or("missing masses")?;
    for i in 0..7 {
        for j in 0..i {
            let difference = std::array::from_fn(|mu| &shifts[i][mu] - &shifts[j][mu]);
            q_squared = q_squared.max(dot(&difference, &difference).abs());
        }
    }
    Ok(
        json!({"method":"Symbolica Matrix<Q>::det/inv/rank and exact Rational arithmetic",
        "scope":"Necessary geometric/positive-F checks only; production guards and every child remain active",
        "six_systems":six,"five_subsets":five,"all_seven_six_systems_regular":six_regular,
        "all_twenty_one_five_subsets_necessary_domain_supported":five_supported,
        "positive_F_lower_bound":lower.to_string(),
        "normalization_q_squared":q_squared.to_string(),
        "domain_accepted":six_regular && five_supported && lower>Rational::from(0)}),
    )
}
