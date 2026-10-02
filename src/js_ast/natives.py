"""
Round-trips V8 natives syntax (e.g. `%OptimizeFunctionOnNextCall(f)`) through the
esprima parser, which does not understand it.

Before parsing, `%Name(` is rewritten into a call of a placeholder function
`__v8_native_Name(`. After code generation the placeholder is turned back into
natives syntax so V8 (run with --allow-natives-syntax) sees the original call.
"""

import re


NATIVE_PREFIX = "__v8_native_"

# A `%` directly followed by a capitalised identifier and an opening bracket
_ENCODE_RE = re.compile(r"%(?=([A-Z][A-Za-z0-9_]*)\s*\()")
_DECODE_RE = re.compile(re.escape(NATIVE_PREFIX) + r"([A-Za-z0-9_]+)")

# Characters which, when they end the code before a `%`, make it a modulo operator
_OPERAND_END = re.compile(r"[A-Za-z0-9_$)\]'\"`]$")
_TRAILING_WORD = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)$")

# Keywords after which an expression (and so natives syntax) starts
_EXPRESSION_KEYWORDS = {
    "return",
    "else",
    "do",
    "typeof",
    "void",
    "delete",
    "case",
    "in",
    "of",
    "new",
    "throw",
    "yield",
    "await",
}
# Keywords whose bracketed condition is followed by a statement
_STATEMENT_KEYWORDS = {"if", "while", "for", "with"}


def _opening_bracket(code: str, close_index: int) -> int:
    """Index of the `(` matching the `)` at close_index, or -1."""
    depth = 0
    for i in range(close_index, -1, -1):
        if code[i] == ")":
            depth += 1
        elif code[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_natives_position(code: str, index: int) -> bool:
    """Whether a `%` at `index` starts natives syntax rather than being a modulo."""
    line_start = code.rfind("\n", 0, index) + 1
    if not code[line_start:index].strip():
        # First thing on its line, i.e. the start of a statement
        return True

    before = code[:index].rstrip()
    if not _OPERAND_END.search(before):
        return True

    word = _TRAILING_WORD.search(before)
    if word and word.group(1) in _EXPRESSION_KEYWORDS:
        return True

    if before.endswith(")"):
        # `if (x) %Foo()`: a statement follows the condition
        open_index = _opening_bracket(before, len(before) - 1)
        keyword = _TRAILING_WORD.search(before[:open_index].rstrip())
        return bool(keyword and keyword.group(1) in _STATEMENT_KEYWORDS)

    return False


def encode_natives(code: str) -> str:
    out: list[str] = []
    last = 0

    for match in _ENCODE_RE.finditer(code):
        start = match.start()
        if not _is_natives_position(code, start):
            continue

        out.append(code[last:start])
        out.append(NATIVE_PREFIX)
        last = start + 1

    out.append(code[last:])
    return "".join(out)


def decode_natives(code: str) -> str:
    return _DECODE_RE.sub(r"%\1", code)


def is_native_name(name: str) -> bool:
    return name.startswith(NATIVE_PREFIX)
