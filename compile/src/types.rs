use oqi_parse::ast;

use crate::classical::{
    ArrayRefShape, ArrayRefTy, ArrayShape, ArrayTy, Duration, DurationUnit, PrimitiveTy, RefAccess,
    ScalarTy, Value, ValueTy, adim, bit_width, bitreg_value, bool_value, complex_value,
    duration_value, float_value, int_value, value_as_usize,
};
use crate::error::{CompileError, ErrorKind, Result, ResultExt};
use crate::resolve::lookup_intrinsic;
use crate::sir::Intrinsic;
use crate::symbol::SymbolTable;

pub use crate::classical::FloatWidth;

/// Fully resolved type — no expression designators, no lifetimes.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Void,
    Classical(ValueTy),
    Stretch,
    /// Single `qubit`.
    Qubit,
    /// `qubit[N]`.
    QubitReg(usize),
    /// Physical qubit reference: `$0`, `$1`.
    PhysicalQubit,
}

impl Type {
    pub fn bool() -> Self {
        PrimitiveTy::Bool.into()
    }

    pub fn bit() -> Self {
        PrimitiveTy::Bit.into()
    }

    pub fn bitreg(width: usize) -> Self {
        PrimitiveTy::BitReg(bit_width(width)).into()
    }

    pub fn int(width: usize, signed: bool) -> Self {
        if signed {
            PrimitiveTy::Int(bit_width(width)).into()
        } else {
            PrimitiveTy::Uint(bit_width(width)).into()
        }
    }

    pub fn float(width: FloatWidth) -> Self {
        PrimitiveTy::Float(width).into()
    }

    pub fn angle(width: usize) -> Self {
        PrimitiveTy::Angle(bit_width(width)).into()
    }

    pub fn complex(width: FloatWidth) -> Self {
        PrimitiveTy::Complex(width).into()
    }

    pub fn duration() -> Self {
        PrimitiveTy::Duration.into()
    }

    pub fn array(element: ScalarTy, dims: Vec<usize>) -> Self {
        ArrayTy::new(
            element,
            ArrayShape::new(dims).expect("compile-generated array shape should be valid"),
        )
        .into()
    }

    pub fn array_of(element: Type, dims: Vec<usize>) -> Self {
        Self::array(
            element
                .scalar_ty()
                .expect("compile-generated array element type should be scalar"),
            dims,
        )
    }

    pub fn array_ref_fixed(element: ScalarTy, dims: Vec<usize>, access: RefAccess) -> Self {
        ArrayRefTy::new(
            element,
            ArrayRefShape::Fixed(
                ArrayShape::new(dims).expect("compile-generated array-ref shape should be valid"),
            ),
            access,
        )
        .into()
    }

    pub fn array_ref_fixed_of(element: Type, dims: Vec<usize>, access: RefAccess) -> Self {
        Self::array_ref_fixed(
            element
                .scalar_ty()
                .expect("compile-generated array-ref element type should be scalar"),
            dims,
            access,
        )
    }

    pub fn array_ref_rank(element: ScalarTy, rank: usize, access: RefAccess) -> Self {
        ArrayRefTy::new(element, ArrayRefShape::Dim(adim(rank)), access).into()
    }

    pub fn array_ref_rank_of(element: Type, rank: usize, access: RefAccess) -> Self {
        Self::array_ref_rank(
            element
                .scalar_ty()
                .expect("compile-generated array-ref element type should be scalar"),
            rank,
            access,
        )
    }

    pub fn value_ty(&self) -> Option<ValueTy> {
        match self {
            Type::Classical(ty) => Some(*ty),
            Type::Stretch => Some(ValueTy::Scalar(PrimitiveTy::Duration)),
            _ => None,
        }
    }

    pub fn scalar_ty(&self) -> Option<ScalarTy> {
        match self.value_ty()? {
            ValueTy::Scalar(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn array_ty(&self) -> Option<ArrayTy> {
        match self.value_ty()? {
            ValueTy::Array(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn array_ref_ty(&self) -> Option<ArrayRefTy> {
        match self.value_ty()? {
            ValueTy::ArrayRef(ty) => Some(ty),
            _ => None,
        }
    }
}

impl From<PrimitiveTy> for Type {
    fn from(value: PrimitiveTy) -> Self {
        Type::Classical(ValueTy::Scalar(value))
    }
}

impl From<ValueTy> for Type {
    fn from(value: ValueTy) -> Self {
        Type::Classical(value)
    }
}

impl From<ArrayTy> for Type {
    fn from(value: ArrayTy) -> Self {
        Type::Classical(ValueTy::Array(value))
    }
}

impl From<ArrayRefTy> for Type {
    fn from(value: ArrayRefTy) -> Self {
        Type::Classical(ValueTy::ArrayRef(value))
    }
}

/// Map a system width to a `FloatWidth`. The OpenQASM spec supports 32 and 64.
pub fn float_width_from_system_width(width: usize) -> FloatWidth {
    match width {
        w if w <= 32 => FloatWidth::F32,
        _ => FloatWidth::F64,
    }
}

/// Options affecting type resolution and lowering.
pub struct CompileOptions {
    /// Source file path, used for include resolution and diagnostics.
    pub source_name: Option<std::path::PathBuf>,
    /// Default width for plain `int`, `uint`, `float`, `complex`, and `angle`.
    pub system_angle_width: usize,
    /// The duration of a single `dt` (device time) unit.
    pub dt: Duration,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            source_name: None,
            system_angle_width: usize::BITS as usize,
            dt: Duration::new(1.0, DurationUnit::Us),
        }
    }
}

/// Evaluate a constant expression to a `usize` value for use as a type designator.
pub fn eval_designator(
    expr: &ast::Expr<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<usize> {
    let cv = eval_const_expr(expr, symbols, options)?;
    value_as_usize(&cv)
        .ok_or_else(|| CompileError::new(ErrorKind::NonConstantDesignator).with_span(expr.span()))
}

/// Evaluate a constant expression, returning a `Value`.
pub fn eval_const_expr(
    expr: &ast::Expr<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Value> {
    match expr {
        ast::Expr::IntLiteral(s, encoding, span) => {
            Ok(int_value(parse_int_literal(s, *encoding).with_span(*span)?))
        }
        ast::Expr::FloatLiteral(s, span) => Ok(float_value(
            parse_float_literal(s).with_span(*span)?,
            FloatWidth::F64,
        )),
        ast::Expr::ImagLiteral(s, span) => {
            let (re, im) = parse_imag_literal(s).with_span(*span)?;
            Ok(complex_value(re, im, FloatWidth::F64))
        }
        ast::Expr::BoolLiteral(val, _) => Ok(bool_value(*val)),
        ast::Expr::BitstringLiteral(s, span) => {
            let (bits, width) = parse_bitstring_literal(s).with_span(*span)?;
            Ok(bitreg_value(bits, width))
        }
        ast::Expr::TimingLiteral(s, span) => Ok(duration_value(
            parse_timing_literal(s, &options.dt).with_span(*span)?,
        )),
        ast::Expr::Paren(inner, _) => eval_const_expr(inner, symbols, options),
        ast::Expr::Ident(ident) => {
            let sym_id = symbols.lookup(ident.name).ok_or_else(|| {
                CompileError::new(ErrorKind::UndefinedName(ident.name.to_string()))
                    .with_span(ident.span)
            })?;
            let sym = symbols.get(sym_id);
            match &sym.const_value {
                Some(cv) => Ok(cv.clone()),
                None => {
                    Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(ident.span))
                }
            }
        }
        ast::Expr::BinOp {
            left,
            op,
            right,
            span,
        } => {
            let lv = eval_const_expr(left, symbols, options)?;
            let rv = eval_const_expr(right, symbols, options)?;
            const_binop(lv, *op, rv, *span)
        }
        ast::Expr::UnaryOp { op, operand, span } => {
            let v = eval_const_expr(operand, symbols, options)?;
            const_unary(*op, v, *span)
        }
        ast::Expr::Call { name, args, span } => {
            const_intrinsic_call(name.name, args, symbols, options, *span)
        }
        other => Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(other.span())),
    }
}

/// Apply a binary operation to two const values using classical ops.
fn const_binop(lv: Value, op: ast::BinOp, rv: Value, span: oqi_lex::Span) -> Result<Value> {
    let result = match op {
        ast::BinOp::Add => lv.add_(rv),
        ast::BinOp::Sub => lv.sub_(rv),
        ast::BinOp::Mul => lv.mul_(rv),
        ast::BinOp::Div => lv.div_(rv),
        ast::BinOp::Mod => lv.rem_(rv),
        ast::BinOp::Pow => lv.pow_(rv),
        ast::BinOp::BitAnd => lv.and_(rv),
        ast::BinOp::BitOr => lv.or_(rv),
        ast::BinOp::BitXor => lv.xor_(rv),
        ast::BinOp::Shl => lv.shl_(rv),
        ast::BinOp::Shr => lv.shr_(rv),
        ast::BinOp::LogAnd => lv.land_(rv),
        ast::BinOp::LogOr => lv.lor_(rv),
        ast::BinOp::Eq => lv.eq_(rv),
        ast::BinOp::Neq => lv.neq_(rv),
        ast::BinOp::Lt => lv.lt_(rv),
        ast::BinOp::Gt => lv.gt_(rv),
        ast::BinOp::Lte => lv.lte_(rv),
        ast::BinOp::Gte => lv.gte_(rv),
    };
    result.map_err(|_| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))
}

/// Apply a unary operation to a const value.
fn const_unary(op: ast::UnOp, v: Value, span: oqi_lex::Span) -> Result<Value> {
    let result = match op {
        ast::UnOp::Neg => v.neg_(),
        ast::UnOp::BitNot => v.not_(),
        ast::UnOp::LogNot => v.lnot_(),
    };
    result.map_err(|_| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))
}

fn const_intrinsic_call(
    name: &str,
    args: &[ast::Expr<'_>],
    symbols: &SymbolTable,
    options: &CompileOptions,
    span: oqi_lex::Span,
) -> Result<Value> {
    let Some(intrinsic) = lookup_intrinsic(name) else {
        return Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(span));
    };
    match intrinsic {
        Intrinsic::Sin => const_unary_intrinsic(args, symbols, options, span, Value::sin_),
        Intrinsic::Cos => const_unary_intrinsic(args, symbols, options, span, Value::cos_),
        Intrinsic::Tan => const_unary_intrinsic(args, symbols, options, span, Value::tan_),
        Intrinsic::Arcsin => const_unary_intrinsic(args, symbols, options, span, Value::arcsin_),
        Intrinsic::Arccos => const_unary_intrinsic(args, symbols, options, span, Value::arccos_),
        Intrinsic::Arctan => const_unary_intrinsic(args, symbols, options, span, Value::arctan_),
        Intrinsic::Exp => const_unary_intrinsic(args, symbols, options, span, Value::exp_),
        Intrinsic::Log => const_unary_intrinsic(args, symbols, options, span, Value::log_),
        Intrinsic::Sqrt => const_unary_intrinsic(args, symbols, options, span, Value::sqrt_),
        Intrinsic::Ceiling => const_unary_intrinsic(args, symbols, options, span, Value::ceiling_),
        Intrinsic::Floor => const_unary_intrinsic(args, symbols, options, span, Value::floor_),
        Intrinsic::Mod => const_binary_intrinsic(args, symbols, options, span, Value::rem_),
        Intrinsic::Popcount => {
            const_unary_intrinsic(args, symbols, options, span, Value::popcount_)
        }
        Intrinsic::Rotl => const_binary_intrinsic(args, symbols, options, span, Value::rotl_),
        Intrinsic::Rotr => const_binary_intrinsic(args, symbols, options, span, Value::rotr_),
        Intrinsic::Real => const_unary_intrinsic(args, symbols, options, span, Value::real_),
        Intrinsic::Imag => const_unary_intrinsic(args, symbols, options, span, Value::imag_),
        Intrinsic::Sizeof => const_sizeof(args, symbols, options, span),
    }
}

fn const_unary_intrinsic(
    args: &[ast::Expr<'_>],
    symbols: &SymbolTable,
    options: &CompileOptions,
    span: oqi_lex::Span,
    op: impl FnOnce(Value) -> oqi_classical::Result<Value>,
) -> Result<Value> {
    let [arg] = args else {
        return Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(span));
    };
    let arg = eval_const_expr(arg, symbols, options)?;
    op(arg).map_err(|_| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))
}

fn const_binary_intrinsic(
    args: &[ast::Expr<'_>],
    symbols: &SymbolTable,
    options: &CompileOptions,
    span: oqi_lex::Span,
    op: impl FnOnce(Value, Value) -> oqi_classical::Result<Value>,
) -> Result<Value> {
    let [lhs, rhs] = args else {
        return Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(span));
    };
    let lhs = eval_const_expr(lhs, symbols, options)?;
    let rhs = eval_const_expr(rhs, symbols, options)?;
    op(lhs, rhs).map_err(|_| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))
}

fn const_sizeof(
    args: &[ast::Expr<'_>],
    symbols: &SymbolTable,
    options: &CompileOptions,
    span: oqi_lex::Span,
) -> Result<Value> {
    let (value_expr, dim_expr) = match args {
        [value] => (value, None),
        [value, dim] => (value, Some(dim)),
        _ => return Err(CompileError::new(ErrorKind::NonConstantExpression).with_span(span)),
    };

    let dim = match dim_expr {
        Some(dim) => value_as_usize(&eval_const_expr(dim, symbols, options)?)
            .ok_or_else(|| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))?,
        None => 0,
    };

    let value_ty = match value_expr {
        ast::Expr::Ident(ident) => {
            let sym_id = symbols.lookup(ident.name).ok_or_else(|| {
                CompileError::new(ErrorKind::UndefinedName(ident.name.to_string()))
                    .with_span(ident.span)
            })?;
            symbols.get(sym_id).ty.value_ty().ok_or_else(|| {
                CompileError::new(ErrorKind::NonConstantExpression).with_span(span)
            })?
        }
        _ => eval_const_expr(value_expr, symbols, options)?.ty(),
    };

    let size = value_ty
        .size(dim)
        .ok_or_else(|| CompileError::new(ErrorKind::NonConstantExpression).with_span(span))?;
    Ok(int_value(size as i128))
}

#[inline]
pub fn parse_int_literal(s: &str, encoding: ast::IntEncoding) -> Result<i128> {
    let (radix, digits) = match encoding {
        ast::IntEncoding::Binary => (2, &s[2..]),
        ast::IntEncoding::Octal => (8, &s[2..]),
        ast::IntEncoding::Hex => (16, &s[2..]),
        ast::IntEncoding::Decimal => (10, s),
    };
    Ok(i128::from_str_radix(&digits.replace('_', ""), radix)
        .map_err(|e| ErrorKind::InvalidLiteral(e.to_string()))?)
}

#[inline]
pub fn parse_float_literal(s: &str) -> Result<f64> {
    Ok(s.replace('_', "")
        .parse::<f64>()
        .map_err(|e| ErrorKind::InvalidLiteral(e.to_string()))?)
}

#[inline]
pub fn parse_imag_literal(s: &str) -> Result<(f64, f64)> {
    let im = parse_float_literal(s.strip_suffix("im").unwrap_or(s).trim_end())?;
    Ok((0.0, im))
}

#[inline]
pub fn parse_bitstring_literal(s: &str) -> Result<(u128, usize)> {
    let s = s.trim_matches('"');
    let mut value = 0u128;
    let mut width = 0usize;
    for (index, ch) in s.chars().enumerate() {
        match ch {
            '1' | '0' => {
                if width >= 128 {
                    return Err(ErrorKind::InvalidLiteral(
                        "bitstring literal exceeds oqi-classical's 128-bit limit".to_string(),
                    )
                    .into());
                }
                if ch == '1' {
                    value |= 1u128 << width;
                }
                width += 1;
            }
            '_' => {}
            other => {
                return Err(ErrorKind::InvalidLiteral(format!(
                    "invalid character '{other}' at position {index}"
                ))
                .into());
            }
        }
    }
    if width == 0 {
        return Err(ErrorKind::InvalidLiteral("Empty".to_string()).into());
    }
    Ok((value, width))
}

#[inline]
pub fn parse_timing_literal(s: &str, dt: &Duration) -> Result<Duration> {
    let unit_start = s.find(|c: char| c.is_alphabetic()).ok_or_else(|| {
        ErrorKind::InvalidLiteral(format!("invalid timing literal: no unit found in '{s}'"))
    })?;
    let (num_str, unit_str) = s.split_at(unit_start);
    let value = num_str
        .replace('_', "")
        .trim()
        .parse::<f64>()
        .map_err(|e| ErrorKind::InvalidLiteral(e.to_string()))?;
    if matches!(unit_str, "dt") {
        return Ok(Duration::new(value * dt.value, dt.unit));
    }
    let unit = unit_str
        .trim()
        .parse::<DurationUnit>()
        .map_err(|e| ErrorKind::InvalidLiteral(e.to_string()))?;
    Ok(Duration::new(value, unit))
}

/// Resolve an AST scalar type to an IR `Type`.
pub fn resolve_scalar_type(
    ty: &ast::ScalarType<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    match ty {
        ast::ScalarType::Bool(_) => Ok(Type::bool()),
        ast::ScalarType::Duration(_) => Ok(Type::duration()),
        ast::ScalarType::Stretch(_) => Ok(Type::Stretch),

        ast::ScalarType::Bit(None, _) => Ok(Type::bit()),
        ast::ScalarType::Bit(Some(expr), _) => {
            let n = eval_designator(expr, symbols, options)?;
            Ok(Type::bitreg(n))
        }

        ast::ScalarType::Int(None, _) => Ok(Type::int(options.system_angle_width, true)),
        ast::ScalarType::Int(Some(expr), _) => {
            let width = eval_designator(expr, symbols, options)?;
            Ok(Type::int(width, true))
        }

        ast::ScalarType::Uint(None, _) => Ok(Type::int(options.system_angle_width, false)),
        ast::ScalarType::Uint(Some(expr), _) => {
            let width = eval_designator(expr, symbols, options)?;
            Ok(Type::int(width, false))
        }

        ast::ScalarType::Float(None, _) => Ok(Type::float(float_width_from_system_width(
            options.system_angle_width,
        ))),
        ast::ScalarType::Float(Some(expr), _) => {
            let width = eval_designator(expr, symbols, options)?;
            Ok(Type::float(float_width_from_system_width(width)))
        }

        ast::ScalarType::Angle(None, _) => Ok(Type::angle(options.system_angle_width)),
        ast::ScalarType::Angle(Some(expr), _) => {
            let width = eval_designator(expr, symbols, options)?;
            Ok(Type::angle(width))
        }

        ast::ScalarType::Complex(None, _) => Ok(Type::complex(float_width_from_system_width(
            options.system_angle_width,
        ))),
        ast::ScalarType::Complex(Some(inner), _) => {
            let inner_ty = resolve_scalar_type(inner, symbols, options)?;
            match inner_ty.scalar_ty() {
                Some(PrimitiveTy::Float(fw)) => Ok(Type::complex(fw)),
                _ => Err(CompileError::new(ErrorKind::Unsupported(
                    "complex designator must be a float type".to_string(),
                ))
                .with_span(ty.span())),
            }
        }
    }
}

/// Resolve an AST type expression to an IR `Type`.
pub fn resolve_type(
    ty: &ast::TypeExpr<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    match ty {
        ast::TypeExpr::Scalar(scalar) => resolve_scalar_type(scalar, symbols, options),
        ast::TypeExpr::Array(arr) => {
            let element = resolve_scalar_type(&arr.element_type, symbols, options)?
                .scalar_ty()
                .expect("resolved scalar type should stay scalar");
            let mut dims = Vec::with_capacity(arr.dimensions.len());
            for dim_expr in &arr.dimensions {
                dims.push(eval_designator(dim_expr, symbols, options)?);
            }
            Ok(Type::array(element, dims))
        }
    }
}

/// Resolve an AST qubit type to an IR `Type`.
pub fn resolve_qubit_type(
    ty: &ast::QubitType<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    match &ty.designator {
        None => Ok(Type::Qubit),
        Some(expr) => {
            let n = eval_designator(expr, symbols, options)?;
            Ok(Type::QubitReg(n))
        }
    }
}

/// Resolve an old-style declaration (`creg`/`qreg`) to an IR `Type`.
pub fn resolve_old_style_type(
    kind: &ast::OldStyleKind,
    designator: Option<&ast::Expr<'_>>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    match kind {
        ast::OldStyleKind::Creg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, symbols, options)?;
                Ok(Type::bitreg(n))
            }
            None => Ok(Type::bit()),
        },
        ast::OldStyleKind::Qreg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, symbols, options)?;
                Ok(Type::QubitReg(n))
            }
            None => Ok(Type::Qubit),
        },
    }
}

/// Resolve an array-ref argument type (subroutine / extern parameters).
pub fn resolve_array_ref_type(
    arr_ref: &ast::ArrayRefType<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    let element = resolve_scalar_type(&arr_ref.element_type, symbols, options)?
        .scalar_ty()
        .expect("resolved scalar type should stay scalar");
    let access = match arr_ref.mutability {
        ast::ArrayRefMut::Readonly => RefAccess::Readonly,
        ast::ArrayRefMut::Mutable => RefAccess::Mutable,
    };
    match &arr_ref.dimensions {
        ast::ArrayRefDims::ExprList(exprs) => {
            let mut fixed = Vec::with_capacity(exprs.len());
            for e in exprs {
                fixed.push(eval_designator(e, symbols, options)?);
            }
            Ok(Type::array_ref_fixed(element, fixed, access))
        }
        ast::ArrayRefDims::Dim(expr) => {
            let rank = eval_designator(expr, symbols, options)?;
            Ok(Type::array_ref_rank(element, rank, access))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{PrimitiveTy, bit_width};

    fn span(start: usize, end: usize) -> oqi_lex::Span {
        oqi_lex::span(start, end)
    }

    fn empty_symbols() -> SymbolTable {
        SymbolTable::new()
    }

    fn default_options() -> CompileOptions {
        CompileOptions {
            source_name: None,
            system_angle_width: 64,
            ..Default::default()
        }
    }

    #[test]
    fn test_parse_int_literal_decimal() {
        let int = parse_int_literal("42", ast::IntEncoding::Decimal).unwrap();
        assert_eq!(int, 42);
    }

    #[test]
    fn test_parse_int_literal_hex() {
        let int = parse_int_literal("0xFF", ast::IntEncoding::Hex).unwrap();
        assert_eq!(int, 255);
    }

    #[test]
    fn test_parse_int_literal_binary() {
        let int = parse_int_literal("0b1010", ast::IntEncoding::Binary).unwrap();
        assert_eq!(int, 10);
    }

    #[test]
    fn test_parse_int_literal_octal() {
        let int = parse_int_literal("0o77", ast::IntEncoding::Octal).unwrap();
        assert_eq!(int, 63);
    }

    #[test]
    fn test_parse_int_literal_with_separators() {
        let int = parse_int_literal("1_000", ast::IntEncoding::Decimal).unwrap();
        assert_eq!(int, 1000);
    }

    #[test]
    fn test_resolve_scalar_bool() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Bool(span(0, 4));
        assert_eq!(resolve_scalar_type(&ty, &sym, &opts).unwrap(), Type::bool());
    }

    #[test]
    fn test_resolve_scalar_bit_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Bit(None, span(0, 3));
        assert_eq!(resolve_scalar_type(&ty, &sym, &opts).unwrap(), Type::bit());
    }

    #[test]
    fn test_resolve_scalar_bit_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(4, 5));
        let ty = ast::ScalarType::Bit(Some(Box::new(expr)), span(0, 6));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::bitreg(4)
        );
    }

    #[test]
    fn test_resolve_scalar_int_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Int(None, span(0, 3));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::int(64, true)
        );
    }

    #[test]
    fn test_resolve_scalar_int_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("8", ast::IntEncoding::Decimal, span(4, 5));
        let ty = ast::ScalarType::Int(Some(Box::new(expr)), span(0, 6));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::int(8, true)
        );
    }

    #[test]
    fn test_resolve_scalar_uint_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("16", ast::IntEncoding::Decimal, span(5, 7));
        let ty = ast::ScalarType::Uint(Some(Box::new(expr)), span(0, 8));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::int(16, false)
        );
    }

    #[test]
    fn test_resolve_scalar_float_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Float(None, span(0, 5));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::float(FloatWidth::F64)
        );
    }

    #[test]
    fn test_resolve_scalar_float_32() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, span(6, 8));
        let ty = ast::ScalarType::Float(Some(Box::new(expr)), span(0, 9));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::float(FloatWidth::F32)
        );
    }

    #[test]
    fn test_resolve_scalar_angle_no_designator() {
        let sym = empty_symbols();
        let opts = CompileOptions {
            source_name: None,
            system_angle_width: 32,
            ..Default::default()
        };
        let ty = ast::ScalarType::Angle(None, span(0, 5));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::angle(32)
        );
    }

    #[test]
    fn test_resolve_scalar_complex_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Complex(None, span(0, 7));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::complex(FloatWidth::F64)
        );
    }

    #[test]
    fn test_resolve_scalar_complex_with_float32() {
        let sym = empty_symbols();
        let opts = default_options();
        let inner_expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, span(14, 16));
        let inner = ast::ScalarType::Float(Some(Box::new(inner_expr)), span(8, 17));
        let ty = ast::ScalarType::Complex(Some(Box::new(inner)), span(0, 18));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::complex(FloatWidth::F32)
        );
    }

    #[test]
    fn test_resolve_scalar_duration() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Duration(span(0, 8));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::duration()
        );
    }

    #[test]
    fn test_resolve_scalar_stretch() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Stretch(span(0, 7));
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Stretch
        );
    }

    #[test]
    fn test_resolve_array_type() {
        let sym = empty_symbols();
        let opts = default_options();
        let elem = ast::ScalarType::Float(
            Some(Box::new(ast::Expr::IntLiteral(
                "64",
                ast::IntEncoding::Decimal,
                span(12, 14),
            ))),
            span(6, 15),
        );
        let dim1 = ast::Expr::IntLiteral("3", ast::IntEncoding::Decimal, span(17, 18));
        let dim2 = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(20, 21));
        let arr = ast::ArrayType {
            element_type: elem,
            dimensions: vec![dim1, dim2],
            span: span(0, 22),
        };
        let ty = ast::TypeExpr::Array(arr);
        assert_eq!(
            resolve_type(&ty, &sym, &opts).unwrap(),
            Type::array(PrimitiveTy::Float(FloatWidth::F64), vec![3, 4])
        );
    }

    #[test]
    fn test_resolve_qubit_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::QubitType {
            designator: None,
            span: span(0, 5),
        };
        assert_eq!(resolve_qubit_type(&ty, &sym, &opts).unwrap(), Type::Qubit);
    }

    #[test]
    fn test_resolve_qubit_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(6, 7));
        let ty = ast::QubitType {
            designator: Some(Box::new(expr)),
            span: span(0, 8),
        };
        assert_eq!(
            resolve_qubit_type(&ty, &sym, &opts).unwrap(),
            Type::QubitReg(4)
        );
    }

    #[test]
    fn test_resolve_old_style_creg() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(5, 6));
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Creg, Some(&expr), &sym, &opts).unwrap(),
            Type::bitreg(4)
        );
    }

    #[test]
    fn test_resolve_old_style_qreg() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(5, 6));
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Qreg, Some(&expr), &sym, &opts).unwrap(),
            Type::QubitReg(4)
        );
    }

    #[test]
    fn test_eval_const_expr_from_symbol() {
        use crate::symbol::SymbolKind;

        let mut sym = SymbolTable::new();
        let id = sym.insert(
            "N".to_string(),
            SymbolKind::Const,
            Type::int(32, false),
            span(0, 5),
        );
        sym.set_const_value(
            id,
            int_value(parse_int_literal("8", ast::IntEncoding::Decimal).unwrap()),
        );

        let expr = ast::Expr::Ident(ast::Ident {
            name: "N",
            span: span(0, 1),
        });
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &sym, &options).unwrap();
        assert_eq!(value_as_usize(&val), Some(8));
    }

    #[test]
    fn test_eval_const_expr_imag_literal() {
        let expr = ast::Expr::ImagLiteral("3 im", span(0, 4));
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Complex(FloatWidth::F64));
                let value = scalar.value().as_complex(FloatWidth::F64).unwrap();
                assert_eq!(value.re, 0.0);
                assert_eq!(value.im, 3.0);
            }
            other => panic!("expected complex literal, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_bitstring_literal() {
        let expr = ast::Expr::BitstringLiteral(r#""0110""#, span(0, 6));
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::BitReg(bit_width(4)));
                assert_eq!(scalar.value().as_bitreg(bit_width(4)), Some(0b0110));
            }
            other => panic!("expected bitstring literal, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_timing_literal() {
        let dt = Duration::new(0.5, DurationUnit::Ns);
        let options = CompileOptions {
            dt,
            ..Default::default()
        };
        let expr = ast::Expr::TimingLiteral("4 dt", span(0, 4));
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Duration);
                assert_eq!(
                    scalar.value().as_duration(),
                    Some(Duration::new(4.0 * dt.value, dt.unit))
                );
            }
            other => panic!("expected timing literal, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_sin() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "sin",
                span: span(0, 3),
            },
            args: vec![ast::Expr::FloatLiteral("0.0", span(4, 7))],
            span: span(0, 8),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(0.0));
            }
            other => panic!("expected float result, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_popcount() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "popcount",
                span: span(0, 8),
            },
            args: vec![ast::Expr::BitstringLiteral(r#""1011""#, span(9, 15))],
            span: span(0, 16),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Uint(bit_width(4)));
                assert_eq!(scalar.value().as_uint(bit_width(4)), Some(3));
            }
            other => panic!("expected uint result, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_rotl() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "rotl",
                span: span(0, 4),
            },
            args: vec![
                ast::Expr::BitstringLiteral(r#""1001""#, span(5, 11)),
                ast::Expr::IntLiteral("1", ast::IntEncoding::Decimal, span(13, 14)),
            ],
            span: span(0, 15),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::BitReg(bit_width(4)));
                assert_eq!(scalar.value().as_bitreg(bit_width(4)), Some(0b0011));
            }
            other => panic!("expected bitreg result, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_sizeof_symbol() {
        use crate::symbol::SymbolKind;

        let mut sym = SymbolTable::new();
        sym.insert(
            "my_array".to_string(),
            SymbolKind::Variable,
            Type::array(PrimitiveTy::Uint(bit_width(8)), vec![2, 3, 4]),
            span(0, 8),
        );

        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "sizeof",
                span: span(0, 6),
            },
            args: vec![
                ast::Expr::Ident(ast::Ident {
                    name: "my_array",
                    span: span(7, 15),
                }),
                ast::Expr::IntLiteral("1", ast::IntEncoding::Decimal, span(17, 18)),
            ],
            span: span(0, 19),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &sym, &options).unwrap();
        assert_eq!(value_as_usize(&val), Some(3));
    }

    #[test]
    fn test_eval_const_expr_intrinsic_real() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "real",
                span: span(0, 4),
            },
            args: vec![ast::Expr::ImagLiteral("3 im", span(5, 9))],
            span: span(0, 10),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(0.0));
            }
            other => panic!("expected float result, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_imag() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "imag",
                span: span(0, 4),
            },
            args: vec![ast::Expr::ImagLiteral("3 im", span(5, 9))],
            span: span(0, 10),
        };
        let options = CompileOptions::default();
        let val = eval_const_expr(&expr, &empty_symbols(), &options).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(3.0));
            }
            other => panic!("expected float result, got {other:?}"),
        }
    }

    #[test]
    fn test_float_width_from_system_width() {
        assert_eq!(float_width_from_system_width(32), FloatWidth::F32);
        assert_eq!(float_width_from_system_width(64), FloatWidth::F64);
        assert_eq!(float_width_from_system_width(16), FloatWidth::F32);
    }
}
