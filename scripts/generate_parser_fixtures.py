#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["antlr4-python3-runtime==4.13.2"]
# ///
"""Generate JSON parser fixtures from .qasm files using the ANTLR4 qasm3Parser."""

import json
import sys
from pathlib import Path

# Add generated parser to path
sys.path.insert(0, str(Path(__file__).parent / "antlr_generated"))

from antlr4 import CommonTokenStream, InputStream
from qasm3Lexer import qasm3Lexer
from qasm3Parser import qasm3Parser

FIXTURES_QASM = Path(__file__).resolve().parent.parent / "fixtures" / "qasm"
FIXTURES_PARSER = Path(__file__).resolve().parent.parent / "fixtures" / "parser"


def tree_to_dict(node, parser) -> dict:
    """Recursively convert an ANTLR parse tree node to a JSON-serializable dict."""
    if hasattr(node, "getRuleIndex"):
        # Rule (interior) node
        rule_name = parser.ruleNames[node.getRuleIndex()]
        # The class name encodes the labeled alternative (e.g. AdditiveExpressionContext)
        cls = type(node).__name__
        default_cls = rule_name[0].upper() + rule_name[1:] + "Context"
        label = cls.removesuffix("Context") if cls != default_cls else None

        result: dict = {"rule": rule_name}
        if label:
            result["label"] = label

        # Character offsets (inclusive, matching ANTLR convention)
        if node.start is not None:
            result["start"] = node.start.start
        if node.stop is not None:
            result["stop"] = node.stop.stop

        children = []
        for child in node.children or []:
            children.append(tree_to_dict(child, parser))
        result["children"] = children
        return result
    else:
        # Terminal (token) node
        tok = node.symbol
        # EOF token has type -1; skip it
        if tok.type == -1:
            return {"token": "EOF", "text": "<EOF>", "start": tok.start, "stop": tok.stop}
        sym = parser.symbolicNames[tok.type]
        if not sym:
            sym = parser.literalNames[tok.type] or str(tok.type)
        return {"token": sym, "text": tok.text, "start": tok.start, "stop": tok.stop}


def parse_file(path: Path) -> dict:
    source = path.read_text()
    input_stream = InputStream(source)
    lexer = qasm3Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = qasm3Parser(stream)
    tree = parser.program()
    return tree_to_dict(tree, parser)


def main():
    FIXTURES_PARSER.mkdir(parents=True, exist_ok=True)

    qasm_files = sorted(FIXTURES_QASM.glob("*.qasm"))
    if not qasm_files:
        print("No .qasm files found in", FIXTURES_QASM, file=sys.stderr)
        sys.exit(1)

    for qasm_path in qasm_files:
        out_path = FIXTURES_PARSER / (qasm_path.stem + ".json")
        tree = parse_file(qasm_path)
        out_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False) + "\n")
        child_count = len(tree.get("children", []))
        print(f"{qasm_path.name} -> {out_path.name}  ({child_count} top-level children)")


if __name__ == "__main__":
    main()
