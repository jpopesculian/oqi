#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["antlr4-python3-runtime==4.13.2"]
# ///
"""Generate JSON lexer fixtures from .qasm files using the ANTLR4 qasm3Lexer."""

import json
import sys
from pathlib import Path

# Add generated lexer to path
sys.path.insert(0, str(Path(__file__).parent / "antlr_generated"))

from antlr4 import CommonTokenStream, InputStream
from qasm3Lexer import qasm3Lexer

FIXTURES_QASM = Path(__file__).resolve().parent.parent / "fixtures" / "qasm"
FIXTURES_LEXER = Path(__file__).resolve().parent.parent / "fixtures" / "lexer"


def lex_file(path: Path) -> list[dict]:
    source = path.read_text()
    input_stream = InputStream(source)
    lexer = qasm3Lexer(input_stream)
    # Get ALL tokens including hidden channel (comments)
    tokens = lexer.getAllTokens()

    result = []
    for tok in tokens:
        token_type = lexer.symbolicNames[tok.type] or lexer.ruleNames[tok.type - 1]
        entry = {
            "type": token_type,
            "text": tok.text,
            "line": tok.line,
            "column": tok.column,
            "start": tok.start,
            "stop": tok.stop,
            "channel": tok.channel,
        }
        result.append(entry)
    return result


def main():
    FIXTURES_LEXER.mkdir(parents=True, exist_ok=True)

    qasm_files = sorted(FIXTURES_QASM.glob("*.qasm"))
    if not qasm_files:
        print("No .qasm files found in", FIXTURES_QASM, file=sys.stderr)
        sys.exit(1)

    for qasm_path in qasm_files:
        out_path = FIXTURES_LEXER / (qasm_path.stem + ".json")
        tokens = lex_file(qasm_path)
        out_path.write_text(json.dumps(tokens, indent=2, ensure_ascii=False) + "\n")
        print(f"{qasm_path.name} -> {out_path.name}  ({len(tokens)} tokens)")


if __name__ == "__main__":
    main()
