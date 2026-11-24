use oqi_classical::ops;
use oqi_lex::Span;
use oqi_parse::ast;

use crate::classical::{
    ArrayDim, ArrayRefShape, ArrayRefTy, ArrayShape, ArrayTy, Duration, DurationUnit, FloatWidth,
    IntWidth, PrimitiveTy, RefAccess, Value, ValueTy, iw,
};
use crate::error::{CompileError, ErrorKind, Result, ResultExt};
use crate::openpulse;
use crate::resolve::Resolver;
use crate::sir::{BinOp, Intrinsic, UnOp};

/// An integer-like or float literal whose width is not yet fixed. Lives only
/// in the compiler: it collapses to a concrete sized type the moment it meets
/// one (a cast, an assignment, a loop variable, a sized binary operand), so it
/// never reaches the bytecode/VM. A value that never meets a sized type is
/// carried at 64 bits ([`Type::value_ty`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsizedType {
    Int,
    Uint,
    Angle,
    Float,
}

/// Fully resolved type — no expression designators, no lifetimes.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Void,
    Classical(ValueTy),
    /// A literal whose width collapses to its surrounding sized type.
    Unsized(UnsizedType),
    Stretch,
    /// Single `qubit`.
    Qubit,
    /// `qubit[N]`.
    QubitReg(usize),
    /// Physical qubit reference: `$0`, `$1`.
    PhysicalQubit,
    /// OpenPulse primitive (`port`, `frame`, `waveform`).
    Openpulse(openpulse::PrimitiveTy),
}

impl Type {
    pub fn value_ty(&self) -> Option<ValueTy> {
        match self {
            Type::Classical(ty) => Some(*ty),
            Type::Stretch => Some(ValueTy::Scalar(PrimitiveTy::Duration)),
            // An uncollapsed literal is carried at 64 bits (`float[64]`
            // for floats).
            Type::Unsized(u) => Some(ValueTy::Scalar(match u {
                UnsizedType::Int => PrimitiveTy::Int(iw(64)),
                UnsizedType::Uint => PrimitiveTy::Uint(iw(64)),
                UnsizedType::Angle => PrimitiveTy::Angle(iw(64)),
                UnsizedType::Float => PrimitiveTy::Float(FloatWidth::F64),
            })),
            _ => None,
        }
    }

    /// True for an unsized literal type that should collapse to context.
    pub fn unsized_ty(&self) -> Option<UnsizedType> {
        match self {
            Type::Unsized(u) => Some(*u),
            _ => None,
        }
    }

    pub fn scalar_ty(&self) -> Option<PrimitiveTy> {
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
pub(crate) fn float_width_from_system_width(width: usize) -> FloatWidth {
    match width {
        w if w <= 32 => FloatWidth::F32,
        _ => FloatWidth::F64,
    }
}

/// Width of plain (undesignated) `int`, `uint`, `float`, `complex` and
/// `angle`. The spec leaves it to "the particular target architecture"
/// (docs/types.rst); the target here is the simulator, not the machine the
/// compiler happens to run on, so the default is a fixed 64 bits on every
/// host (native and wasm32 alike).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SystemWidth {
    /// 32-bit `int`/`uint`/`angle`, `float[32]`.
    B32,
    /// 64-bit `int`/`uint`/`angle`, `float[64]`.
    #[default]
    B64,
}

impl SystemWidth {
    /// The width of plain `int`, `uint` and `angle`.
    pub fn iw(self) -> IntWidth {
        match self {
            SystemWidth::B32 => iw(32),
            SystemWidth::B64 => iw(64),
        }
    }

    /// The width of plain `float` and `complex`.
    pub fn fw(self) -> FloatWidth {
        match self {
            SystemWidth::B32 => FloatWidth::F32,
            SystemWidth::B64 => FloatWidth::F64,
        }
    }
}

/// Options affecting type resolution and lowering. Set them through
/// [`Compiler`](crate::Compiler), or start from `CompileOptions::default()`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CompileOptions {
    /// Source file path, used for include resolution and diagnostics.
    pub source_name: Option<std::path::PathBuf>,
    /// Default width for plain `int`, `uint`, `float`, `complex`, and `angle`.
    pub system_width: SystemWidth,
    /// The duration of a single `dt` (device time) unit.
    pub dt: Duration,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            source_name: None,
            system_width: SystemWidth::default(),
            dt: Duration::new(1.0, DurationUnit::Us),
        }
    }
}

/// `value` as a `usize`, if it is a non-negative integer that fits.
pub(crate) fn value_as_usize(value: &Value) -> Option<usize> {
    let Value::Scalar(scalar) = value else {
        return None;
    };
    let scalar = scalar.clone().cast(PrimitiveTy::Int(iw(128))).ok()?;
    usize::try_from(scalar.value().as_int(iw(128))?).ok()
}

/// Evaluate a constant expression to a `usize` value for use as a type designator.
pub(crate) fn eval_designator(expr: &ast::Expr<'_>, scope: &Resolver) -> Result<usize> {
    let cv = eval_const_expr(expr, scope)?;
    value_as_usize(&cv)
        .ok_or_else(|| CompileError::at(ErrorKind::NonConstantDesignator, expr.span()))
}

/// Evaluate a constant expression, returning a `Value`. Names are resolved
/// in the scope `scope` is in.
pub(crate) fn eval_const_expr(expr: &ast::Expr<'_>, scope: &Resolver) -> Result<Value> {
    match expr {
        ast::Expr::IntLiteral(s, encoding, span) => Ok(Value::int(
            parse_int_literal(s, *encoding).with_span(*span)?,
            iw(128),
        )),
        ast::Expr::FloatLiteral(s, span) => Ok(Value::float(
            parse_float_literal(s).with_span(*span)?,
            FloatWidth::F64,
        )),
        ast::Expr::ImaginaryLiteral(s, span) => {
            let (re, im) = parse_imag_literal(s).with_span(*span)?;
            Ok(Value::complex(re, im, FloatWidth::F64))
        }
        ast::Expr::BoolLiteral(val, _) => Ok(Value::bit(*val)),
        ast::Expr::BitstringLiteral(s, span) => {
            let (bits, width) = parse_bitstring_literal(s).with_span(*span)?;
            Ok(Value::bitreg_u128(bits, width as u32))
        }
        ast::Expr::TimingLiteral(s, span) => Ok(Value::from(
            parse_timing_literal(s, &scope.options().dt).with_span(*span)?,
        )),
        ast::Expr::Paren(inner, _) => eval_const_expr(inner, scope),
        ast::Expr::Ident(ident) => {
            let sym = scope.symbols().get(scope.resolve(ident.name, ident.span)?);
            match &sym.const_value {
                Some(cv) => Ok(cv.clone()),
                None => Err(CompileError::at(
                    ErrorKind::NonConstantExpression,
                    ident.span,
                )),
            }
        }
        ast::Expr::Binary {
            left,
            op,
            right,
            span,
        } => {
            let lv = eval_const_expr(left, scope)?;
            let rv = eval_const_expr(right, scope)?;
            bin_op(*op)
                .apply(lv, rv)
                .map_err(|_| CompileError::at(ErrorKind::NonConstantExpression, *span))
        }
        ast::Expr::Unary { op, operand, span } => {
            let v = eval_const_expr(operand, scope)?;
            un_op(*op)
                .apply(v)
                .map_err(|_| CompileError::at(ErrorKind::NonConstantExpression, *span))
        }
        ast::Expr::Call { name, args, span } => const_intrinsic_call(name.name, args, scope, *span),
        ast::Expr::Cast { ty, operand, span } => {
            let target = resolve_type(ty, scope)?;
            let target_vty = target
                .value_ty()
                .ok_or_else(|| CompileError::at(ErrorKind::NonConstantExpression, *span))?;
            let v = eval_const_expr(operand, scope)?;
            v.cast(target_vty)
                .map_err(|_| CompileError::at(ErrorKind::NonConstantExpression, *span))
        }
        other => Err(CompileError::at(
            ErrorKind::NonConstantExpression,
            other.span(),
        )),
    }
}

/// The classical operator an AST binary operator denotes.
pub(crate) fn bin_op(op: ast::BinOp) -> BinOp {
    match op {
        ast::BinOp::Add => BinOp::Add,
        ast::BinOp::Sub => BinOp::Sub,
        ast::BinOp::Mul => BinOp::Mul,
        ast::BinOp::Div => BinOp::Div,
        ast::BinOp::Mod => BinOp::Mod,
        ast::BinOp::Pow => BinOp::Pow,
        ast::BinOp::BitAnd => BinOp::BitAnd,
        ast::BinOp::BitOr => BinOp::BitOr,
        ast::BinOp::BitXor => BinOp::BitXor,
        ast::BinOp::Shl => BinOp::Shl,
        ast::BinOp::Shr => BinOp::Shr,
        ast::BinOp::LogAnd => BinOp::LogAnd,
        ast::BinOp::LogOr => BinOp::LogOr,
        ast::BinOp::Eq => BinOp::Eq,
        ast::BinOp::Neq => BinOp::Neq,
        ast::BinOp::Lt => BinOp::Lt,
        ast::BinOp::Gt => BinOp::Gt,
        ast::BinOp::Lte => BinOp::Lte,
        ast::BinOp::Gte => BinOp::Gte,
    }
}

/// The classical operator an AST unary operator denotes.
pub(crate) fn un_op(op: ast::UnOp) -> UnOp {
    match op {
        ast::UnOp::Neg => UnOp::Neg,
        ast::UnOp::BitNot => UnOp::BitNot,
        ast::UnOp::LogNot => UnOp::LogNot,
    }
}

fn const_intrinsic_call(
    name: &str,
    args: &[ast::Expr<'_>],
    scope: &Resolver,
    span: Span,
) -> Result<Value> {
    let non_const = || CompileError::at(ErrorKind::NonConstantExpression, span);
    let intrinsic = Intrinsic::lookup(name).ok_or_else(non_const)?;
    if intrinsic == Intrinsic::Sizeof {
        return const_sizeof(args, scope, span);
    }
    let args = args
        .iter()
        .map(|arg| eval_const_expr(arg, scope))
        .collect::<Result<Vec<_>>>()?;
    intrinsic
        .apply(args)
        .ok_or_else(non_const)?
        .map_err(|_| non_const())
}

fn const_sizeof(args: &[ast::Expr<'_>], scope: &Resolver, span: oqi_lex::Span) -> Result<Value> {
    let (value_expr, dim_expr) = match args {
        [value] => (value, None),
        [value, dim] => (value, Some(dim)),
        _ => return Err(CompileError::at(ErrorKind::NonConstantExpression, span)),
    };

    let dim_value = dim_expr
        .map(|dim| eval_const_expr(dim, scope))
        .transpose()?;

    if let ast::Expr::Ident(ident) = value_expr {
        let value_ty = scope
            .symbols()
            .get(scope.resolve(ident.name, ident.span)?)
            .ty
            .value_ty()
            .ok_or_else(|| CompileError::at(ErrorKind::NonConstantExpression, span))?;
        return const_sizeof_ty(value_ty, dim_value, span);
    }

    let value_ty = eval_const_expr(value_expr, scope)?.ty();
    const_sizeof_ty(value_ty, dim_value, span)
}

fn const_sizeof_ty(value_ty: ValueTy, dim: Option<Value>, span: Span) -> Result<Value> {
    ops::sizeof(value_ty, dim).map_err(|_| CompileError::at(ErrorKind::NonConstantExpression, span))
}

/// A type error from the classical layer (an out-of-range width or rank).
pub(crate) fn invalid_type(err: oqi_classical::Error, span: Span) -> CompileError {
    CompileError::at(ErrorKind::InvalidType(err.to_string()), span)
}

/// The width an `int`/`uint`/`angle` designator asks for.
fn int_width(designator: &ast::Expr<'_>, width: usize) -> Result<IntWidth> {
    IntWidth::new(u32::try_from(width).unwrap_or(u32::MAX))
        .map_err(|e| invalid_type(e, designator.span()))
}

#[inline]
pub(crate) fn parse_int_literal(s: &str, encoding: ast::IntEncoding) -> Result<i128> {
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
pub(crate) fn parse_float_literal(s: &str) -> Result<f64> {
    Ok(s.replace('_', "")
        .parse::<f64>()
        .map_err(|e| ErrorKind::InvalidLiteral(e.to_string()))?)
}

#[inline]
pub(crate) fn parse_imag_literal(s: &str) -> Result<(f64, f64)> {
    let im = parse_float_literal(s.strip_suffix("im").unwrap_or(s).trim_end())?;
    Ok((0.0, im))
}

#[inline]
pub(crate) fn parse_bitstring_literal(s: &str) -> Result<(u128, usize)> {
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
                // MSB first: the leftmost character is the most significant
                // bit (docs/types.rst: `"00001111"` has decimal value 15).
                value = (value << 1) | u128::from(ch == '1');
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
pub(crate) fn parse_timing_literal(s: &str, dt: &Duration) -> Result<Duration> {
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

/// Parse the duration of one `dt` unit from a host option ("0.5ns"):
/// timing-literal syntax, except the `dt` unit itself is rejected
/// (self-referential) and the value must be positive.
pub fn parse_dt(spec: &str) -> Result<Duration> {
    let invalid = |msg: String| CompileError::new(ErrorKind::InvalidLiteral(msg));
    if spec.trim_end().ends_with("dt") {
        return Err(invalid(
            "`dt` cannot itself be given in `dt` units".to_string(),
        ));
    }
    let d = parse_timing_literal(spec, &CompileOptions::default().dt).map_err(|_| {
        invalid(format!(
            "`dt`: `{spec}` is not a duration literal (e.g. \"0.5ns\")"
        ))
    })?;
    if d.value.is_nan() || d.value <= 0.0 {
        return Err(invalid("`dt` must be a positive duration".to_string()));
    }
    Ok(d)
}

/// Resolve an AST scalar type to an IR `Type`.
pub(crate) fn resolve_scalar_type(ty: &ast::ScalarType<'_>, scope: &Resolver) -> Result<Type> {
    use ast::{ScalarKind, SizedKind};

    let system = &scope.options().system_width;
    let classical = match &ty.kind {
        ScalarKind::Bool => ValueTy::bool(),
        ScalarKind::Duration => ValueTy::duration(),
        ScalarKind::Stretch => return Ok(Type::Stretch),

        ScalarKind::Sized(kind, None) => match kind {
            SizedKind::Bit => ValueTy::bit(),
            SizedKind::Int => ValueTy::int(system.iw()),
            SizedKind::Uint => ValueTy::uint(system.iw()),
            SizedKind::Float => ValueTy::float(system.fw()),
            SizedKind::Angle => ValueTy::angle(system.iw()),
        },
        ScalarKind::Sized(kind, Some(expr)) => {
            let width = eval_designator(expr, scope)?;
            match kind {
                SizedKind::Bit => ValueTy::bitreg(width as u32),
                SizedKind::Int => ValueTy::int(int_width(expr, width)?),
                SizedKind::Uint => ValueTy::uint(int_width(expr, width)?),
                SizedKind::Float => ValueTy::float(float_width_from_system_width(width)),
                SizedKind::Angle => ValueTy::angle(int_width(expr, width)?),
            }
        }

        ScalarKind::Complex(None) => ValueTy::complex(system.fw()),
        ScalarKind::Complex(Some(inner)) => match resolve_scalar_type(inner, scope)?.scalar_ty() {
            Some(PrimitiveTy::Float(fw)) => ValueTy::complex(fw),
            _ => {
                return Err(CompileError::at(
                    ErrorKind::Unsupported("complex designator must be a float type".to_string()),
                    ty.span,
                ));
            }
        },

        ScalarKind::Waveform => {
            return Ok(Type::Openpulse(openpulse::PrimitiveTy::Waveform(
                system.fw(),
            )));
        }
        ScalarKind::Port => return Ok(Type::Openpulse(openpulse::PrimitiveTy::Port)),
        ScalarKind::Frame => {
            return Ok(Type::Openpulse(openpulse::PrimitiveTy::Frame(
                system.fw(),
                system.iw(),
            )));
        }
    };
    Ok(Type::Classical(classical))
}

/// Resolve an AST type expression to an IR `Type`.
pub(crate) fn resolve_type(ty: &ast::TypeExpr<'_>, scope: &Resolver) -> Result<Type> {
    match ty {
        ast::TypeExpr::Scalar(scalar) => resolve_scalar_type(scalar, scope),
        ast::TypeExpr::Array(arr) => {
            let element = resolve_scalar_type(&arr.element_type, scope)?
                .scalar_ty()
                .expect("resolved scalar type should stay scalar");
            let mut dims = Vec::with_capacity(arr.dimensions.len());
            for dim_expr in &arr.dimensions {
                dims.push(eval_designator(dim_expr, scope)?);
            }
            let shape = ArrayShape::new(dims).map_err(|e| invalid_type(e, arr.span))?;
            Ok(Type::Classical(ValueTy::array(element, shape)))
        }
    }
}

/// Resolve an AST qubit type to an IR `Type`.
pub(crate) fn resolve_qubit_type(ty: &ast::QubitType<'_>, scope: &Resolver) -> Result<Type> {
    match &ty.designator {
        None => Ok(Type::Qubit),
        Some(expr) => {
            let n = eval_designator(expr, scope)?;
            Ok(Type::QubitReg(n))
        }
    }
}

/// Resolve an old-style declaration (`creg`/`qreg`) to an IR `Type`.
pub(crate) fn resolve_old_style_type(
    kind: &ast::OldStyleKind,
    designator: Option<&ast::Expr<'_>>,
    scope: &Resolver,
) -> Result<Type> {
    match kind {
        ast::OldStyleKind::Creg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, scope)?;
                Ok(Type::Classical(ValueTy::bitreg(n as u32)))
            }
            None => Ok(Type::Classical(ValueTy::bit())),
        },
        ast::OldStyleKind::Qreg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, scope)?;
                Ok(Type::QubitReg(n))
            }
            None => Ok(Type::Qubit),
        },
    }
}

/// Resolve an array-ref argument type (subroutine / extern parameters).
pub(crate) fn resolve_array_ref_type(
    arr_ref: &ast::ArrayRefType<'_>,
    scope: &Resolver,
) -> Result<Type> {
    let element = resolve_scalar_type(&arr_ref.element_type, scope)?
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
                fixed.push(eval_designator(e, scope)?);
            }
            let shape = ArrayShape::new(fixed).map_err(|e| invalid_type(e, arr_ref.span))?;
            Ok(Type::Classical(ValueTy::array_ref(
                element,
                ArrayRefShape::Fixed(shape),
                access,
            )))
        }
        ast::ArrayRefDims::Dim(expr) => {
            let rank = eval_designator(expr, scope)?;
            let rank = ArrayDim::new(rank).map_err(|e| invalid_type(e, expr.span()))?;
            Ok(Type::Classical(ValueTy::array_ref(
                element,
                ArrayRefShape::Dim(rank),
                access,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{PrimitiveTy, ashape, iw};

    fn span(start: usize, end: usize) -> oqi_lex::Span {
        oqi_lex::Span::new(start, end)
    }

    fn scalar<'a>(kind: ast::ScalarKind<'a>, span: oqi_lex::Span) -> ast::ScalarType<'a> {
        ast::ScalarType { kind, span }
    }

    fn sized<'a>(
        kind: ast::SizedKind,
        designator: Option<ast::Expr<'a>>,
        span: oqi_lex::Span,
    ) -> ast::ScalarType<'a> {
        scalar(ast::ScalarKind::Sized(kind, designator.map(Box::new)), span)
    }

    /// The global scope, with only the built-in names in it.
    fn global_scope() -> Resolver<'static> {
        let options = CompileOptions {
            source_name: None,
            system_width: SystemWidth::B64,
            ..Default::default()
        };
        Resolver::new(&crate::resolve::DefaultIncludeResolver, options)
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
        let scope = global_scope();
        let ty = scalar(ast::ScalarKind::Bool, span(0, 4));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::bool())
        );
    }

    #[test]
    fn test_resolve_scalar_bit_no_designator() {
        let scope = global_scope();
        let ty = sized(ast::SizedKind::Bit, None, span(0, 3));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::bit())
        );
    }

    #[test]
    fn test_resolve_scalar_bit_with_designator() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(4, 5));
        let ty = sized(ast::SizedKind::Bit, Some(expr), span(0, 6));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::bitreg(4))
        );
    }

    #[test]
    fn test_resolve_scalar_int_no_designator() {
        let scope = global_scope();
        let ty = sized(ast::SizedKind::Int, None, span(0, 3));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::int(iw(64)))
        );
    }

    #[test]
    fn test_resolve_scalar_int_with_designator() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("8", ast::IntEncoding::Decimal, span(4, 5));
        let ty = sized(ast::SizedKind::Int, Some(expr), span(0, 6));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::int(iw(8)))
        );
    }

    #[test]
    fn test_resolve_scalar_uint_with_designator() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("16", ast::IntEncoding::Decimal, span(5, 7));
        let ty = sized(ast::SizedKind::Uint, Some(expr), span(0, 8));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::uint(iw(16)))
        );
    }

    #[test]
    fn test_resolve_scalar_float_no_designator() {
        let scope = global_scope();
        let ty = sized(ast::SizedKind::Float, None, span(0, 5));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::float(FloatWidth::F64))
        );
    }

    #[test]
    fn test_resolve_scalar_float_32() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, span(6, 8));
        let ty = sized(ast::SizedKind::Float, Some(expr), span(0, 9));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::float(FloatWidth::F32))
        );
    }

    #[test]
    fn test_resolve_scalar_angle_no_designator() {
        let options = CompileOptions {
            source_name: None,
            system_width: SystemWidth::B32,
            ..Default::default()
        };
        let scope = Resolver::new(&crate::resolve::DefaultIncludeResolver, options);
        let ty = sized(ast::SizedKind::Angle, None, span(0, 5));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::angle(iw(32)))
        );
    }

    #[test]
    fn test_resolve_scalar_complex_no_designator() {
        let scope = global_scope();
        let ty = scalar(ast::ScalarKind::Complex(None), span(0, 7));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::complex(FloatWidth::F64))
        );
    }

    #[test]
    fn test_resolve_scalar_complex_with_float32() {
        let scope = global_scope();
        let inner_expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, span(14, 16));
        let inner = sized(ast::SizedKind::Float, Some(inner_expr), span(8, 17));
        let ty = scalar(ast::ScalarKind::Complex(Some(Box::new(inner))), span(0, 18));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::complex(FloatWidth::F32))
        );
    }

    #[test]
    fn test_resolve_scalar_duration() {
        let scope = global_scope();
        let ty = scalar(ast::ScalarKind::Duration, span(0, 8));
        assert_eq!(
            resolve_scalar_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::duration())
        );
    }

    #[test]
    fn test_resolve_scalar_stretch() {
        let scope = global_scope();
        let ty = scalar(ast::ScalarKind::Stretch, span(0, 7));
        assert_eq!(resolve_scalar_type(&ty, &scope).unwrap(), Type::Stretch);
    }

    #[test]
    fn test_resolve_array_type() {
        let scope = global_scope();
        let elem = sized(
            ast::SizedKind::Float,
            Some(ast::Expr::IntLiteral(
                "64",
                ast::IntEncoding::Decimal,
                span(12, 14),
            )),
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
            resolve_type(&ty, &scope).unwrap(),
            Type::Classical(ValueTy::array(
                PrimitiveTy::Float(FloatWidth::F64),
                ashape(vec![3, 4])
            ))
        );
    }

    #[test]
    fn test_resolve_qubit_no_designator() {
        let scope = global_scope();
        let ty = ast::QubitType {
            designator: None,
            span: span(0, 5),
        };
        assert_eq!(resolve_qubit_type(&ty, &scope).unwrap(), Type::Qubit);
    }

    #[test]
    fn test_resolve_qubit_with_designator() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(6, 7));
        let ty = ast::QubitType {
            designator: Some(Box::new(expr)),
            span: span(0, 8),
        };
        assert_eq!(resolve_qubit_type(&ty, &scope).unwrap(), Type::QubitReg(4));
    }

    #[test]
    fn test_resolve_old_style_creg() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(5, 6));
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Creg, Some(&expr), &scope).unwrap(),
            Type::Classical(ValueTy::bitreg(4))
        );
    }

    #[test]
    fn test_resolve_old_style_qreg() {
        let scope = global_scope();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, span(5, 6));
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Qreg, Some(&expr), &scope).unwrap(),
            Type::QubitReg(4)
        );
    }

    #[test]
    fn test_eval_const_expr_from_symbol() {
        use crate::symbol::SymbolKind;

        let mut scope = global_scope();
        let ty = Type::Classical(ValueTy::uint(iw(32)));
        let id = scope
            .declare("N", SymbolKind::Const, ty, span(0, 5))
            .unwrap();
        scope.symbols_mut().set_const_value(
            id,
            Value::int(
                parse_int_literal("8", ast::IntEncoding::Decimal).unwrap(),
                iw(128),
            ),
        );

        let expr = ast::Expr::Ident(ast::Ident {
            name: "N",
            span: span(0, 1),
        });
        let val = eval_const_expr(&expr, &scope).unwrap();
        assert_eq!(value_as_usize(&val), Some(8));
    }

    #[test]
    fn test_eval_const_expr_imag_literal() {
        let expr = ast::Expr::ImaginaryLiteral("3 im", span(0, 4));
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
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
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::BitReg(4));
                assert_eq!(
                    scalar.value().as_bitreg(4).map(|r| r.as_u128()),
                    Some(0b0110_u128)
                );
            }
            other => panic!("expected bitstring literal, got {other:?}"),
        }
    }

    /// The leftmost character is the most significant bit: the spec's
    /// `"00001111"` example has decimal value 15 (docs/types.rst).
    #[test]
    fn test_bitstring_literal_is_msb_first() {
        assert_eq!(parse_bitstring_literal(r#""00001111""#).unwrap(), (15, 8));
        assert_eq!(parse_bitstring_literal(r#""1000""#).unwrap(), (8, 4));
        assert_eq!(parse_bitstring_literal(r#""0001_0001""#).unwrap(), (17, 8));
    }

    #[test]
    fn test_eval_const_expr_timing_literal() {
        let dt = Duration::new(0.5, DurationUnit::Ns);
        let options = CompileOptions {
            dt,
            ..Default::default()
        };
        let scope = Resolver::new(&crate::resolve::DefaultIncludeResolver, options);
        let expr = ast::Expr::TimingLiteral("4 dt", span(0, 4));
        let val = eval_const_expr(&expr, &scope).unwrap();
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
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
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
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Uint(iw(4)));
                assert_eq!(scalar.value().as_uint(iw(4)), Some(3));
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
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::BitReg(4));
                assert_eq!(
                    scalar.value().as_bitreg(4).map(|r| r.as_u128()),
                    Some(0b0011_u128)
                );
            }
            other => panic!("expected bitreg result, got {other:?}"),
        }
    }

    #[test]
    fn test_eval_const_expr_intrinsic_sizeof_symbol() {
        use crate::symbol::SymbolKind;

        let mut scope = global_scope();
        let ty = Type::Classical(ValueTy::array(
            PrimitiveTy::Uint(iw(8)),
            ashape(vec![2, 3, 4]),
        ));
        scope
            .declare("my_array", SymbolKind::Variable, ty, span(0, 8))
            .unwrap();

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
        let val = eval_const_expr(&expr, &scope).unwrap();
        assert_eq!(value_as_usize(&val), Some(3));
    }

    #[test]
    fn test_eval_const_expr_intrinsic_sizeof_value_expr() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "sizeof",
                span: span(0, 6),
            },
            args: vec![ast::Expr::BitstringLiteral(r#""1010""#, span(7, 13))],
            span: span(0, 14),
        };
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
        assert_eq!(value_as_usize(&val), Some(4));
    }

    #[test]
    fn test_eval_const_expr_intrinsic_real() {
        let expr = ast::Expr::Call {
            name: ast::Ident {
                name: "real",
                span: span(0, 4),
            },
            args: vec![ast::Expr::ImaginaryLiteral("3 im", span(5, 9))],
            span: span(0, 10),
        };
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
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
            args: vec![ast::Expr::ImaginaryLiteral("3 im", span(5, 9))],
            span: span(0, 10),
        };
        let val = eval_const_expr(&expr, &global_scope()).unwrap();
        match val {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(3.0));
            }
            other => panic!("expected float result, got {other:?}"),
        }
    }
}
