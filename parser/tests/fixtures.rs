use oqi_parser::ast::*;
use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// ANTLR CST deserialization
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct Cst {
    rule: Option<String>,
    label: Option<String>,
    token: Option<String>,
    text: Option<String>,
    #[allow(dead_code)]
    start: Option<usize>,
    #[allow(dead_code)]
    stop: Option<usize>,
    #[serde(default)]
    children: Vec<Cst>,
}

impl Cst {
    fn find_rule(&self, name: &str) -> Option<&Cst> {
        self.children
            .iter()
            .find(|c| c.rule.as_deref() == Some(name))
    }
    fn find_rules(&self, name: &str) -> Vec<&Cst> {
        self.children
            .iter()
            .filter(|c| c.rule.as_deref() == Some(name))
            .collect()
    }
    fn find_token(&self, name: &str) -> Option<&Cst> {
        self.children
            .iter()
            .find(|c| c.token.as_deref() == Some(name))
    }
}

// ---------------------------------------------------------------------------
// Simplified tree for structural comparison
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct S {
    kind: String,
    children: Vec<S>,
}

impl S {
    fn leaf(kind: impl Into<String>) -> Self {
        S {
            kind: kind.into(),
            children: vec![],
        }
    }
    fn node(kind: impl Into<String>, children: Vec<S>) -> Self {
        S {
            kind: kind.into(),
            children,
        }
    }
}

impl std::fmt::Display for S {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write(node: &S, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
            for _ in 0..depth {
                f.write_str("  ")?;
            }
            writeln!(f, "{}", node.kind)?;
            for child in &node.children {
                write(child, f, depth + 1)?;
            }
            Ok(())
        }
        write(self, f, 0)
    }
}

// ---------------------------------------------------------------------------
// CST -> S
// ---------------------------------------------------------------------------

fn cst_program(cst: &Cst) -> S {
    assert_eq!(cst.rule.as_deref(), Some("program"));
    let mut children = vec![];
    if cst.find_rule("version").is_some() {
        children.push(S::leaf("Version"));
    }
    for c in &cst.children {
        if c.rule.as_deref() == Some("statementOrScope") {
            children.push(cst_stmt_or_scope(c));
        }
    }
    S::node("Program", children)
}

fn cst_stmt_or_scope(cst: &Cst) -> S {
    for c in &cst.children {
        match c.rule.as_deref() {
            Some("statement") => return cst_statement(c),
            Some("scope") => return cst_scope(c),
            _ => {}
        }
    }
    panic!("statementOrScope with no statement or scope child")
}

fn cst_scope(cst: &Cst) -> S {
    S::node(
        "Scope",
        cst.children
            .iter()
            .filter(|c| c.rule.as_deref() == Some("statementOrScope"))
            .map(cst_stmt_or_scope)
            .collect(),
    )
}

fn cst_statement(cst: &Cst) -> S {
    // statement = pragma | annotation* actualStatementRule
    for c in &cst.children {
        if c.rule.is_some() && c.rule.as_deref() != Some("annotation") {
            return cst_stmt_rule(c);
        }
    }
    panic!("statement with no non-annotation rule child")
}

fn cst_stmt_rule(cst: &Cst) -> S {
    let rule = cst.rule.as_deref().unwrap_or("");
    match rule {
        "pragma" => S::leaf("Pragma"),
        "calibrationGrammarStatement" => S::leaf("CalibrationGrammar"),
        "includeStatement" => S::leaf("Include"),
        "breakStatement" => S::leaf("Break"),
        "continueStatement" => S::leaf("Continue"),
        "endStatement" => S::leaf("End"),

        "forStatement" => {
            let body = cst
                .find_rule("statementOrScope")
                .expect("for: no body");
            S::node("For", vec![cst_stmt_or_scope(body)])
        }

        "ifStatement" => {
            let mut children = vec![];
            if let Some(e) = cst.find_rule("expression") {
                children.push(cst_expr(e));
            }
            for body in cst.find_rules("statementOrScope") {
                children.push(cst_stmt_or_scope(body));
            }
            S::node("If", children)
        }

        "returnStatement" => S::leaf("Return"),

        "whileStatement" => {
            let mut children = vec![];
            if let Some(e) = cst.find_rule("expression") {
                children.push(cst_expr(e));
            }
            if let Some(body) = cst.find_rule("statementOrScope") {
                children.push(cst_stmt_or_scope(body));
            }
            S::node("While", children)
        }

        "switchStatement" => S::node(
            "Switch",
            cst.find_rules("switchCaseItem")
                .into_iter()
                .map(|c| cst_scope(c.find_rule("scope").expect("switchCaseItem: no scope")))
                .collect(),
        ),

        "barrierStatement" => S::leaf("Barrier"),

        "boxStatement" => S::node(
            "Box",
            vec![cst_scope(
                cst.find_rule("scope").expect("box: no scope"),
            )],
        ),

        "delayStatement" => S::leaf("Delay"),
        "nopStatement" => S::leaf("Nop"),
        "gateCallStatement" => S::leaf("GateCall"),
        "measureArrowAssignmentStatement" => S::leaf("MeasureArrow"),
        "resetStatement" => S::leaf("Reset"),

        "aliasDeclarationStatement" => S::leaf("Alias"),
        "classicalDeclarationStatement" => S::leaf("ClassicalDecl"),
        "constDeclarationStatement" => S::leaf("ConstDecl"),
        "ioDeclarationStatement" => S::leaf("IoDecl"),
        "oldStyleDeclarationStatement" => S::leaf("OldStyleDecl"),
        "quantumDeclarationStatement" => S::leaf("QuantumDecl"),

        "defStatement" => S::node(
            "Def",
            vec![cst_scope(
                cst.find_rule("scope").expect("def: no scope"),
            )],
        ),
        "externStatement" => S::leaf("Extern"),
        "gateStatement" => S::node(
            "Gate",
            vec![cst_scope(
                cst.find_rule("scope").expect("gate: no scope"),
            )],
        ),

        "assignmentStatement" => S::leaf("Assignment"),

        "expressionStatement" => {
            let mut children = vec![];
            if let Some(e) = cst.find_rule("expression") {
                children.push(cst_expr(e));
            }
            S::node("ExprStmt", children)
        }

        "calStatement" => S::leaf("Cal"),
        "defcalStatement" => S::leaf("Defcal"),

        _ => panic!("unknown CST statement rule: {rule}"),
    }
}

fn cst_expr(cst: &Cst) -> S {
    let label = cst.label.as_deref().unwrap_or("");
    match label {
        "LiteralExpression" => {
            let tok = cst
                .children
                .first()
                .expect("LiteralExpression: no child");
            let kind = match tok.token.as_deref().unwrap_or("") {
                "Identifier" => "Ident",
                "DecimalIntegerLiteral" | "BinaryIntegerLiteral"
                | "OctalIntegerLiteral" | "HexIntegerLiteral" => "IntLiteral",
                "FloatLiteral" => "FloatLiteral",
                "ImaginaryLiteral" => "ImagLiteral",
                "BooleanLiteral" => "BoolLiteral",
                "BitstringLiteral" => "BitstringLiteral",
                "TimingLiteral" => "TimingLiteral",
                "HardwareQubit" => "HardwareQubit",
                other => panic!("unknown literal token type: {other}"),
            };
            S::leaf(kind)
        }

        "ParenthesisExpression" => {
            let inner = cst.find_rule("expression").expect("paren: no inner");
            S::node("Paren", vec![cst_expr(inner)])
        }

        "IndexExpression" => {
            let base = cst.find_rule("expression").expect("index: no base");
            S::node("Index", vec![cst_expr(base)])
        }

        "PowerExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(**)", exprs)
        }

        "UnaryExpression" => {
            let op = cst
                .children
                .iter()
                .find(|c| {
                    matches!(
                        c.token.as_deref(),
                        Some("MINUS" | "TILDE" | "EXCLAMATION_POINT")
                    )
                })
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            let operand = cst
                .find_rule("expression")
                .map(cst_expr)
                .expect("unary: no operand");
            S::node(format!("UnaryOp({op})"), vec![operand])
        }

        "MultiplicativeExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            let op = cst
                .children
                .iter()
                .find(|c| {
                    matches!(
                        c.token.as_deref(),
                        Some("ASTERISK" | "SLASH" | "PERCENT")
                    )
                })
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            S::node(format!("BinOp({op})"), exprs)
        }

        "AdditiveExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            let op = cst
                .children
                .iter()
                .find(|c| matches!(c.token.as_deref(), Some("PLUS" | "MINUS")))
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            S::node(format!("BinOp({op})"), exprs)
        }

        "BitshiftExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            let op = cst
                .find_token("BitshiftOperator")
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            S::node(format!("BinOp({op})"), exprs)
        }

        "ComparisonExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            let op = cst
                .find_token("ComparisonOperator")
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            S::node(format!("BinOp({op})"), exprs)
        }

        "EqualityExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            let op = cst
                .find_token("EqualityOperator")
                .map(|c| c.text.as_deref().unwrap_or("?"))
                .unwrap_or("?");
            S::node(format!("BinOp({op})"), exprs)
        }

        "BitwiseAndExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(&)", exprs)
        }

        "BitwiseXorExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(^)", exprs)
        }

        "BitwiseOrExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(|)", exprs)
        }

        "LogicalAndExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(&&)", exprs)
        }

        "LogicalOrExpression" => {
            let exprs: Vec<_> = cst
                .find_rules("expression")
                .into_iter()
                .map(cst_expr)
                .collect();
            S::node("BinOp(||)", exprs)
        }

        "CastExpression" => {
            // The last expression child is the operand being cast.
            let exprs = cst.find_rules("expression");
            let operand = cst_expr(exprs.last().expect("cast: no operand"));
            S::node("Cast", vec![operand])
        }

        "DurationofExpression" => {
            let scope = cst
                .find_rule("scope")
                .expect("durationof: no scope");
            S::node("DurationOf", vec![cst_scope(scope)])
        }

        "CallExpression" => {
            let args: Vec<_> = cst
                .find_rule("expressionList")
                .map(|el| {
                    el.find_rules("expression")
                        .into_iter()
                        .map(cst_expr)
                        .collect()
                })
                .unwrap_or_default();
            S::node("Call", args)
        }

        "" => {
            // No label — pass through to child expression if present.
            if let Some(e) = cst.find_rule("expression") {
                cst_expr(e)
            } else {
                panic!("expression with no label and no expression child")
            }
        }

        _ => panic!("unknown expression label: {label}"),
    }
}

// ---------------------------------------------------------------------------
// AST -> S
// ---------------------------------------------------------------------------

fn ast_program(prog: &Program) -> S {
    let mut children = vec![];
    if prog.version.is_some() {
        children.push(S::leaf("Version"));
    }
    for item in &prog.body {
        children.push(ast_stmt_or_scope(item));
    }
    S::node("Program", children)
}

fn ast_stmt_or_scope(sos: &StmtOrScope) -> S {
    match sos {
        StmtOrScope::Stmt(stmt) => ast_stmt(stmt),
        StmtOrScope::Scope(scope) => ast_scope(scope),
    }
}

fn ast_scope(scope: &Scope) -> S {
    S::node(
        "Scope",
        scope.body.iter().map(ast_stmt_or_scope).collect(),
    )
}

fn ast_stmt(stmt: &Stmt) -> S {
    match &stmt.kind {
        StmtKind::Pragma(_) => S::leaf("Pragma"),
        StmtKind::CalibrationGrammar(_) => S::leaf("CalibrationGrammar"),
        StmtKind::Include(_) => S::leaf("Include"),
        StmtKind::Break => S::leaf("Break"),
        StmtKind::Continue => S::leaf("Continue"),
        StmtKind::End => S::leaf("End"),

        StmtKind::For { body, .. } => {
            S::node("For", vec![ast_stmt_or_scope(body)])
        }

        StmtKind::If {
            condition,
            then_body,
            else_body,
        } => {
            let mut children = vec![ast_expr(condition)];
            children.push(ast_stmt_or_scope(then_body));
            if let Some(eb) = else_body {
                children.push(ast_stmt_or_scope(eb));
            }
            S::node("If", children)
        }

        StmtKind::Return(_) => S::leaf("Return"),

        StmtKind::While { condition, body } => {
            S::node("While", vec![ast_expr(condition), ast_stmt_or_scope(body)])
        }

        StmtKind::Switch { cases, .. } => S::node(
            "Switch",
            cases
                .iter()
                .map(|c| {
                    let scope = match c {
                        SwitchCase::Case(_, s) | SwitchCase::Default(s) => s,
                    };
                    ast_scope(scope)
                })
                .collect(),
        ),

        StmtKind::Barrier(_) => S::leaf("Barrier"),
        StmtKind::Box { body, .. } => S::node("Box", vec![ast_scope(body)]),
        StmtKind::Delay { .. } => S::leaf("Delay"),
        StmtKind::Nop(_) => S::leaf("Nop"),
        StmtKind::GateCall { .. } => S::leaf("GateCall"),
        StmtKind::MeasureArrow { .. } => S::leaf("MeasureArrow"),
        StmtKind::Reset(_) => S::leaf("Reset"),

        StmtKind::Alias { .. } => S::leaf("Alias"),
        StmtKind::ClassicalDecl { .. } => S::leaf("ClassicalDecl"),
        StmtKind::ConstDecl { .. } => S::leaf("ConstDecl"),
        StmtKind::IoDecl { .. } => S::leaf("IoDecl"),
        StmtKind::OldStyleDecl { .. } => S::leaf("OldStyleDecl"),
        StmtKind::QuantumDecl { .. } => S::leaf("QuantumDecl"),

        StmtKind::Def { body, .. } => S::node("Def", vec![ast_scope(body)]),
        StmtKind::Extern { .. } => S::leaf("Extern"),
        StmtKind::Gate { body, .. } => S::node("Gate", vec![ast_scope(body)]),

        StmtKind::Assignment { .. } => S::leaf("Assignment"),
        StmtKind::Expr(e) => S::node("ExprStmt", vec![ast_expr(e)]),

        StmtKind::Cal(_) => S::leaf("Cal"),
        StmtKind::Defcal { .. } => S::leaf("Defcal"),
    }
}

fn ast_expr(expr: &Expr) -> S {
    match expr {
        Expr::Ident(_) => S::leaf("Ident"),
        Expr::HardwareQubit(_, _) => S::leaf("HardwareQubit"),
        Expr::IntLiteral(_, _) => S::leaf("IntLiteral"),
        Expr::FloatLiteral(_, _) => S::leaf("FloatLiteral"),
        Expr::ImagLiteral(_, _) => S::leaf("ImagLiteral"),
        Expr::BoolLiteral(_, _) => S::leaf("BoolLiteral"),
        Expr::BitstringLiteral(_, _) => S::leaf("BitstringLiteral"),
        Expr::TimingLiteral(_, _) => S::leaf("TimingLiteral"),

        Expr::Paren(inner, _) => S::node("Paren", vec![ast_expr(inner)]),

        Expr::BinOp {
            left, op, right, ..
        } => {
            let sym = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Mod => "%",
                BinOp::Pow => "**",
                BinOp::BitAnd => "&",
                BinOp::BitOr => "|",
                BinOp::BitXor => "^",
                BinOp::Shl => "<<",
                BinOp::Shr => ">>",
                BinOp::LogAnd => "&&",
                BinOp::LogOr => "||",
                BinOp::Eq => "==",
                BinOp::Neq => "!=",
                BinOp::Lt => "<",
                BinOp::Gt => ">",
                BinOp::Lte => "<=",
                BinOp::Gte => ">=",
            };
            S::node(
                format!("BinOp({sym})"),
                vec![ast_expr(left), ast_expr(right)],
            )
        }

        Expr::UnaryOp { op, operand, .. } => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::BitNot => "~",
                UnOp::LogNot => "!",
            };
            S::node(format!("UnaryOp({sym})"), vec![ast_expr(operand)])
        }

        Expr::Index { expr: base, .. } => {
            S::node("Index", vec![ast_expr(base)])
        }

        Expr::Call { args, .. } => {
            S::node("Call", args.iter().map(ast_expr).collect())
        }

        Expr::Cast { operand, .. } => {
            S::node("Cast", vec![ast_expr(operand)])
        }

        Expr::DurationOf { scope, .. } => {
            S::node("DurationOf", vec![ast_scope(scope)])
        }
    }
}

// ---------------------------------------------------------------------------
// Tree comparison
// ---------------------------------------------------------------------------

fn compare(expected: &S, actual: &S, path: &str) {
    assert_eq!(
        expected.kind,
        actual.kind,
        "at {path}: kind mismatch\n\nantlr subtree:\n{expected}\nour subtree:\n{actual}",
    );
    assert_eq!(
        expected.children.len(),
        actual.children.len(),
        "at {path}/{}: child count mismatch\n  antlr: {:?}\n  ours:  {:?}",
        expected.kind,
        expected.children.iter().map(|c| &c.kind).collect::<Vec<_>>(),
        actual.children.iter().map(|c| &c.kind).collect::<Vec<_>>(),
    );
    for (i, (e, a)) in expected
        .children
        .iter()
        .zip(actual.children.iter())
        .enumerate()
    {
        compare(e, a, &format!("{path}/{}.{i}", expected.kind));
    }
}

// ---------------------------------------------------------------------------
// Fixture runner
// ---------------------------------------------------------------------------

fn check_fixture(name: &str) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let qasm_path = base.join("fixtures/qasm").join(format!("{name}.qasm"));
    let json_path = base
        .join("fixtures/parser")
        .join(format!("{name}.json"));

    let source = std::fs::read_to_string(&qasm_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", qasm_path.display()));

    let cst: Cst = serde_json::from_str(
        &std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_path.display())),
    )
    .unwrap();

    let program = oqi_parser::parse(&source)
        .unwrap_or_else(|e| panic!("{name}: parse error: {e}"));

    let expected = cst_program(&cst);
    let actual = ast_program(&program);
    compare(&expected, &actual, name);
}

macro_rules! fixture_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            check_fixture(stringify!($name));
        }
    };
}

fixture_test!(adder);
fixture_test!(alignment);
fixture_test!(arrays);
fixture_test!(cphase);
fixture_test!(dd);
fixture_test!(defcal);
fixture_test!(gateteleport);
fixture_test!(inverseqft1);
fixture_test!(inverseqft2);
fixture_test!(ipe);
fixture_test!(msd);
fixture_test!(qec);
fixture_test!(qft);
fixture_test!(qpt);
fixture_test!(rb);
fixture_test!(rus);
fixture_test!(scqec);
fixture_test!(t1);
fixture_test!(teleport);
fixture_test!(varteleport);
fixture_test!(vqe);
