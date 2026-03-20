use oqi_parse::ast;

use crate::error::{CompileError, ErrorKind, Result};
use crate::symbol::SymbolTable;
use crate::value::ConstValue;

/// Resolved floating-point width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatWidth {
    F32,
    F64,
}

/// Fully resolved type — no expression designators, no lifetimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,
    Bool,
    /// Single `bit`.
    Bit,
    /// `bit[N]`.
    BitReg(u32),
    /// `int[N]` (signed=true) or `uint[N]` (signed=false).
    Int {
        width: u32,
        signed: bool,
    },
    /// `float[32]` or `float[64]`.
    Float(FloatWidth),
    /// `angle[N]`. Plain `angle` and gate parameters use the system width.
    Angle(u32),
    /// `complex[float[W]]`.
    Complex(FloatWidth),
    Duration,
    Stretch,
    Array {
        element: Box<Type>,
        dims: Vec<u32>,
    },
    ArrayRef {
        element: Box<Type>,
        dims: ArrayRefDims,
        access: ArrayAccess,
    },
    /// Single `qubit`.
    Qubit,
    /// `qubit[N]`.
    QubitReg(u32),
    /// Physical qubit reference: `$0`, `$1`.
    PhysicalQubit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayRefDims {
    Fixed(Vec<u32>),
    /// `#dim = N`
    Rank(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayAccess {
    Readonly,
    Mutable,
}

/// Map a system width to a `FloatWidth`. The OpenQASM spec supports 32 and 64.
pub fn float_width_from_system_width(width: u32) -> FloatWidth {
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
    pub system_angle_width: u32,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            source_name: None,
            system_angle_width: usize::BITS,
        }
    }
}

/// Evaluate a constant expression to a `u32` value for use as a type designator.
///
/// This is a minimal evaluator for Phase 1 that handles integer literals.
/// It will be expanded into the full `eval_const_expr` in later phases.
pub fn eval_designator(expr: &ast::Expr<'_>, symbols: &SymbolTable) -> Result<u32> {
    let cv = eval_const_expr(expr, symbols)?;
    cv.as_u32().ok_or_else(|| CompileError {
        kind: ErrorKind::NonConstantDesignator,
        span: expr.span(),
    })
}

/// Evaluate a constant expression, returning a `ConstValue`.
///
/// Handles: integer literals, const variable references, bool literals.
/// Will be extended in later phases with arithmetic, intrinsics, etc.
pub fn eval_const_expr(expr: &ast::Expr<'_>, symbols: &SymbolTable) -> Result<ConstValue> {
    match expr {
        ast::Expr::IntLiteral(s, encoding, span) => {
            let val = parse_int_literal(s, *encoding).ok_or_else(|| CompileError {
                kind: ErrorKind::NonConstantDesignator,
                span: span.clone(),
            })?;
            Ok(ConstValue::Uint(val))
        }
        ast::Expr::BoolLiteral(val, _) => Ok(ConstValue::Bool(*val)),
        ast::Expr::Ident(ident) => {
            let sym_id = symbols.lookup(ident.name).ok_or_else(|| CompileError {
                kind: ErrorKind::UndefinedName(ident.name.to_string()),
                span: ident.span.clone(),
            })?;
            let sym = symbols.get(sym_id);
            match &sym.const_value {
                Some(cv) => Ok(cv.clone()),
                None => Err(CompileError {
                    kind: ErrorKind::NonConstantExpression,
                    span: ident.span.clone(),
                }),
            }
        }
        other => Err(CompileError {
            kind: ErrorKind::NonConstantExpression,
            span: other.span(),
        }),
    }
}

/// Parse an integer literal string (supports `0b`, `0o`, `0x`, decimal, and `_` separators).
pub fn parse_int_literal(s: &str, encoding: ast::IntEncoding) -> Option<awint::Awi> {
    use core::num::NonZeroUsize;

    let (radix, bits_per_digit, digits) = match encoding {
        ast::IntEncoding::Binary => (2u8, 1, &s[2..]),
        ast::IntEncoding::Octal => (8, 3, &s[2..]),
        ast::IntEncoding::Hex => (16, 4, &s[2..]),
        ast::IntEncoding::Decimal => (10, 0, s),
    };

    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    if clean.is_empty() {
        return None;
    }

    if encoding == ast::IntEncoding::Decimal {
        let mut out = awint::Awi::from_u128(clean.parse().ok()?);
        out.shrink_to_msb();
        Some(out)
    } else {
        let bw = NonZeroUsize::new(clean.len() * bits_per_digit)?;
        awint::Awi::from_str_radix(None, &clean, radix, bw).ok()
    }
}

/// Resolve an AST scalar type to an IR `Type`.
pub fn resolve_scalar_type(
    ty: &ast::ScalarType<'_>,
    symbols: &SymbolTable,
    options: &CompileOptions,
) -> Result<Type> {
    match ty {
        ast::ScalarType::Bool(_) => Ok(Type::Bool),
        ast::ScalarType::Duration(_) => Ok(Type::Duration),
        ast::ScalarType::Stretch(_) => Ok(Type::Stretch),

        ast::ScalarType::Bit(None, _) => Ok(Type::Bit),
        ast::ScalarType::Bit(Some(expr), _) => {
            let n = eval_designator(expr, symbols)?;
            Ok(Type::BitReg(n))
        }

        ast::ScalarType::Int(None, _) => Ok(Type::Int {
            width: options.system_angle_width,
            signed: true,
        }),
        ast::ScalarType::Int(Some(expr), _) => {
            let width = eval_designator(expr, symbols)?;
            Ok(Type::Int {
                width,
                signed: true,
            })
        }

        ast::ScalarType::Uint(None, _) => Ok(Type::Int {
            width: options.system_angle_width,
            signed: false,
        }),
        ast::ScalarType::Uint(Some(expr), _) => {
            let width = eval_designator(expr, symbols)?;
            Ok(Type::Int {
                width,
                signed: false,
            })
        }

        ast::ScalarType::Float(None, _) => Ok(Type::Float(float_width_from_system_width(
            options.system_angle_width,
        ))),
        ast::ScalarType::Float(Some(expr), _) => {
            let width = eval_designator(expr, symbols)?;
            Ok(Type::Float(float_width_from_system_width(width)))
        }

        ast::ScalarType::Angle(None, _) => Ok(Type::Angle(options.system_angle_width)),
        ast::ScalarType::Angle(Some(expr), _) => {
            let width = eval_designator(expr, symbols)?;
            Ok(Type::Angle(width))
        }

        ast::ScalarType::Complex(None, _) => Ok(Type::Complex(float_width_from_system_width(
            options.system_angle_width,
        ))),
        ast::ScalarType::Complex(Some(inner), _) => {
            let inner_ty = resolve_scalar_type(inner, symbols, options)?;
            match inner_ty {
                Type::Float(fw) => Ok(Type::Complex(fw)),
                _ => Err(CompileError {
                    kind: ErrorKind::Unsupported(
                        "complex designator must be a float type".to_string(),
                    ),
                    span: ty.span().clone(),
                }),
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
            let element = resolve_scalar_type(&arr.element_type, symbols, options)?;
            let mut dims = Vec::with_capacity(arr.dimensions.len());
            for dim_expr in &arr.dimensions {
                dims.push(eval_designator(dim_expr, symbols)?);
            }
            Ok(Type::Array {
                element: Box::new(element),
                dims,
            })
        }
    }
}

/// Resolve an AST qubit type to an IR `Type`.
pub fn resolve_qubit_type(ty: &ast::QubitType<'_>, symbols: &SymbolTable) -> Result<Type> {
    match &ty.designator {
        None => Ok(Type::Qubit),
        Some(expr) => {
            let n = eval_designator(expr, symbols)?;
            Ok(Type::QubitReg(n))
        }
    }
}

/// Resolve an old-style declaration (`creg`/`qreg`) to an IR `Type`.
pub fn resolve_old_style_type(
    kind: &ast::OldStyleKind,
    designator: Option<&ast::Expr<'_>>,
    symbols: &SymbolTable,
) -> Result<Type> {
    match kind {
        ast::OldStyleKind::Creg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, symbols)?;
                Ok(Type::BitReg(n))
            }
            None => Ok(Type::Bit),
        },
        ast::OldStyleKind::Qreg => match designator {
            Some(expr) => {
                let n = eval_designator(expr, symbols)?;
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
    let element = resolve_scalar_type(&arr_ref.element_type, symbols, options)?;
    let access = match arr_ref.mutability {
        ast::ArrayRefMut::Readonly => ArrayAccess::Readonly,
        ast::ArrayRefMut::Mutable => ArrayAccess::Mutable,
    };
    let dims = match &arr_ref.dimensions {
        ast::ArrayRefDims::ExprList(exprs) => {
            let mut fixed = Vec::with_capacity(exprs.len());
            for e in exprs {
                fixed.push(eval_designator(e, symbols)?);
            }
            ArrayRefDims::Fixed(fixed)
        }
        ast::ArrayRefDims::Dim(expr) => {
            let rank = eval_designator(expr, symbols)?;
            ArrayRefDims::Rank(rank)
        }
    };
    Ok(Type::ArrayRef {
        element: Box::new(element),
        dims,
        access,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_symbols() -> SymbolTable {
        SymbolTable::new()
    }

    fn default_options() -> CompileOptions {
        CompileOptions {
            source_name: None,
            system_angle_width: 64,
        }
    }

    #[test]
    fn test_parse_int_literal_decimal() {
        let awi = parse_int_literal("42", ast::IntEncoding::Decimal).unwrap();
        assert_eq!(awi.bw(), 6); // 42 fits in 6 bits
    }

    #[test]
    fn test_parse_int_literal_hex() {
        let awi = parse_int_literal("0xFF", ast::IntEncoding::Hex).unwrap();
        assert_eq!(awi.bw(), 8);
    }

    #[test]
    fn test_parse_int_literal_binary() {
        let awi = parse_int_literal("0b1010", ast::IntEncoding::Binary).unwrap();
        assert_eq!(awi.bw(), 4);
    }

    #[test]
    fn test_parse_int_literal_octal() {
        let awi = parse_int_literal("0o77", ast::IntEncoding::Octal).unwrap();
        assert_eq!(awi.bw(), 6);
    }

    #[test]
    fn test_parse_int_literal_with_separators() {
        let awi = parse_int_literal("1_000", ast::IntEncoding::Decimal).unwrap();
        assert_eq!(awi.bw(), 10);
    }

    #[test]
    fn test_resolve_scalar_bool() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Bool(0..4);
        assert_eq!(resolve_scalar_type(&ty, &sym, &opts).unwrap(), Type::Bool);
    }

    #[test]
    fn test_resolve_scalar_bit_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Bit(None, 0..3);
        assert_eq!(resolve_scalar_type(&ty, &sym, &opts).unwrap(), Type::Bit);
    }

    #[test]
    fn test_resolve_scalar_bit_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, 4..5);
        let ty = ast::ScalarType::Bit(Some(Box::new(expr)), 0..6);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::BitReg(4)
        );
    }

    #[test]
    fn test_resolve_scalar_int_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Int(None, 0..3);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Int {
                width: 64,
                signed: true
            }
        );
    }

    #[test]
    fn test_resolve_scalar_int_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("8", ast::IntEncoding::Decimal, 4..5);
        let ty = ast::ScalarType::Int(Some(Box::new(expr)), 0..6);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Int {
                width: 8,
                signed: true
            }
        );
    }

    #[test]
    fn test_resolve_scalar_uint_with_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("16", ast::IntEncoding::Decimal, 5..7);
        let ty = ast::ScalarType::Uint(Some(Box::new(expr)), 0..8);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Int {
                width: 16,
                signed: false
            }
        );
    }

    #[test]
    fn test_resolve_scalar_float_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Float(None, 0..5);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Float(FloatWidth::F64)
        );
    }

    #[test]
    fn test_resolve_scalar_float_32() {
        let sym = empty_symbols();
        let opts = default_options();
        let expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, 6..8);
        let ty = ast::ScalarType::Float(Some(Box::new(expr)), 0..9);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Float(FloatWidth::F32)
        );
    }

    #[test]
    fn test_resolve_scalar_angle_no_designator() {
        let sym = empty_symbols();
        let opts = CompileOptions {
            source_name: None,
            system_angle_width: 32,
        };
        let ty = ast::ScalarType::Angle(None, 0..5);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Angle(32)
        );
    }

    #[test]
    fn test_resolve_scalar_complex_no_designator() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Complex(None, 0..7);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Complex(FloatWidth::F64)
        );
    }

    #[test]
    fn test_resolve_scalar_complex_with_float32() {
        let sym = empty_symbols();
        let opts = default_options();
        let inner_expr = ast::Expr::IntLiteral("32", ast::IntEncoding::Decimal, 14..16);
        let inner = ast::ScalarType::Float(Some(Box::new(inner_expr)), 8..17);
        let ty = ast::ScalarType::Complex(Some(Box::new(inner)), 0..18);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Complex(FloatWidth::F32)
        );
    }

    #[test]
    fn test_resolve_scalar_duration() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Duration(0..8);
        assert_eq!(
            resolve_scalar_type(&ty, &sym, &opts).unwrap(),
            Type::Duration
        );
    }

    #[test]
    fn test_resolve_scalar_stretch() {
        let sym = empty_symbols();
        let opts = default_options();
        let ty = ast::ScalarType::Stretch(0..7);
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
                12..14,
            ))),
            6..15,
        );
        let dim1 = ast::Expr::IntLiteral("3", ast::IntEncoding::Decimal, 17..18);
        let dim2 = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, 20..21);
        let arr = ast::ArrayType {
            element_type: elem,
            dimensions: vec![dim1, dim2],
            span: 0..22,
        };
        let ty = ast::TypeExpr::Array(arr);
        assert_eq!(
            resolve_type(&ty, &sym, &opts).unwrap(),
            Type::Array {
                element: Box::new(Type::Float(FloatWidth::F64)),
                dims: vec![3, 4],
            }
        );
    }

    #[test]
    fn test_resolve_qubit_no_designator() {
        let sym = empty_symbols();
        let ty = ast::QubitType {
            designator: None,
            span: 0..5,
        };
        assert_eq!(resolve_qubit_type(&ty, &sym).unwrap(), Type::Qubit);
    }

    #[test]
    fn test_resolve_qubit_with_designator() {
        let sym = empty_symbols();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, 6..7);
        let ty = ast::QubitType {
            designator: Some(Box::new(expr)),
            span: 0..8,
        };
        assert_eq!(resolve_qubit_type(&ty, &sym).unwrap(), Type::QubitReg(4));
    }

    #[test]
    fn test_resolve_old_style_creg() {
        let sym = empty_symbols();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, 5..6);
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Creg, Some(&expr), &sym).unwrap(),
            Type::BitReg(4)
        );
    }

    #[test]
    fn test_resolve_old_style_qreg() {
        let sym = empty_symbols();
        let expr = ast::Expr::IntLiteral("4", ast::IntEncoding::Decimal, 5..6);
        assert_eq!(
            resolve_old_style_type(&ast::OldStyleKind::Qreg, Some(&expr), &sym).unwrap(),
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
            Type::Int {
                width: 32,
                signed: false,
            },
            0..5,
        );
        sym.set_const_value(
            id,
            ConstValue::Uint(parse_int_literal("8", ast::IntEncoding::Decimal).unwrap()),
        );

        let expr = ast::Expr::Ident(ast::Ident {
            name: "N",
            span: 0..1,
        });
        let val = eval_const_expr(&expr, &sym).unwrap();
        assert_eq!(val.as_u32(), Some(8));
    }

    #[test]
    fn test_float_width_from_system_width() {
        assert_eq!(float_width_from_system_width(32), FloatWidth::F32);
        assert_eq!(float_width_from_system_width(64), FloatWidth::F64);
        assert_eq!(float_width_from_system_width(16), FloatWidth::F32);
    }
}
