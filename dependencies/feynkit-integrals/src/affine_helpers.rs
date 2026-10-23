// Exact affine helpers extracted unchanged from FeynKit.

impl AffineFamily {
    fn exact_polynomial(
        expression: &impl AtomCore,
        variables: &[Atom],
    ) -> Result<ExactPolynomial, IntegralFamilyError> {
        let variables = variables
            .iter()
            .cloned()
            .map(PolyVariable::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(IntegralFamilyError::InvalidBasis)?;
        let field = GaussianField::complex(Q);
        expression
            .try_to_rational_polynomial(&field, &field, None)
            .map_err(|e| IntegralFamilyError::InvalidBasis(e.to_string()))?
            .to_polynomial(&variables, false)
            .map_err(|e| IntegralFamilyError::InvalidBasis(e.into()))
    }

    fn affine_coefficients(
        expression: &impl AtomCore,
        variables: &[Atom],
    ) -> Result<Vec<FamilyCoefficient>, IntegralFamilyError> {
        let polynomial = Self::exact_polynomial(expression, variables)?;
        let field = GaussianField::complex(Q);
        let mut row =
            vec![RationalPolynomial::new(&field, Default::default()); variables.len() + 1];
        for term in &polynomial {
            let mut powers = term.exponents.iter().enumerate().filter(|(_, p)| **p != 0);
            let column = match (powers.next(), powers.next()) {
                (None, None) => variables.len(),
                (Some((index, 1)), None) => index,
                _ => {
                    return Err(IntegralFamilyError::InvalidBasis(
                        "expected affine expressions".into(),
                    ));
                }
            };
            row[column] = term.coefficient.clone();
        }
        Ok(row)
    }

    fn affine_system(
        expressions: &[impl AtomCore],
        variables: &[Atom],
    ) -> Result<(CoefficientMatrix, CoefficientMatrix), IntegralFamilyError> {
        let mut entries: Vec<FamilyCoefficient> = Vec::new();
        for expression in expressions {
            entries.extend(Self::affine_coefficients(expression, variables)?);
        }
        // Every entry must use the same variable ordering for exact arithmetic.
        if let Some((first, rest)) = entries.split_first_mut() {
            for _ in 0..2 {
                for entry in &mut *rest {
                    first.unify_variables(entry);
                }
            }
        }
        let field = RationalPolynomialField::new(GaussianField::complex(Q));
        let mut matrix = Vec::new();
        let mut rhs = Vec::new();
        for row in entries.chunks(variables.len() + 1) {
            matrix.extend_from_slice(&row[..variables.len()]);
            rhs.push(-row[variables.len()].clone());
        }
        Ok((
            Matrix::from_linear(
                matrix,
                expressions.len() as u32,
                variables.len() as u32,
                field.clone(),
            )
            .map_err(IntegralFamilyError::InvalidBasis)?,
            Matrix::new_vec(rhs, field),
        ))
    }

    fn rank_of(&self, denominators: &[Atom]) -> Result<usize, IntegralFamilyError> {
        if denominators.is_empty() {
            return Ok(0);
        }
        let (matrix, constants) = Self::affine_system(denominators, &self.scalar_products)?;
        // Native rational-polynomial conversion treats functions and powers as
        // external atoms. Reject any hidden loop dependence in those atoms too.
        for coefficient in matrix
            .row_iter()
            .flatten()
            .chain(constants.row_iter().flatten())
        {
            self.require_external_coefficient(&coefficient.to_expression())?;
        }
        Ok(matrix.rank())
    }
}
