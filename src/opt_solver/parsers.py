"""Readers for free-format MPS and CPLEX LP files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .types import LPProblem, RowSense, Sense

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN_RE = re.compile(
    rf"\s*(?:"
    rf"(?P<number>{_NUMBER})"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_.$]*)"
    r"|(?P<op>[+\-<>]=?|=)"
    r"|(?P<star>\*)"
    r"|(?P<lparen>\()"
    r"|(?P<rparen>\))"
    r")"
)
_COMPARISON_RE = re.compile(r"(<=|>=|=)")


def read_lp(path: str | Path) -> LPProblem:
    """Read a linear program in CPLEX LP format.

    Integer, binary and semi-continuous variables are rejected because they
    are outside the scope of this project.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()

    sense: Sense = "min"
    section: str | None = None
    objective_chunks: list[str] = []
    constraint_chunks: list[str] = []
    bound_lines: list[str] = []
    pending_constraint = ""

    def flush_constraint() -> None:
        nonlocal pending_constraint
        if pending_constraint.strip():
            constraint_chunks.append(pending_constraint.strip())
            pending_constraint = ""

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("\\"):
            continue

        min_match = re.match(r"^(minimize|minimum|min)\b(.*)$", line, re.IGNORECASE)
        max_match = re.match(r"^(maximize|maximum|max)\b(.*)$", line, re.IGNORECASE)
        subject_match = re.match(
            r"^(subject\s+to|such\s+that|s\.t\.|st)(?:\s+|$)(.*)$", line, re.IGNORECASE
        )
        bounds_match = re.match(r"^(bounds|bound)\b(.*)$", line, re.IGNORECASE)
        integer_match = re.match(
            r"^(general|generals|integer|integers|binary|binaries|semi-continuous|sos)\b",
            line,
            re.IGNORECASE,
        )
        end_match = re.match(r"^end\b", line, re.IGNORECASE)

        if min_match:
            flush_constraint()
            sense = "min"
            section = "objective"
            rest = min_match.group(2).strip()
            if rest:
                objective_chunks.append(rest)
        elif max_match:
            flush_constraint()
            sense = "max"
            section = "objective"
            rest = max_match.group(2).strip()
            if rest:
                objective_chunks.append(rest)
        elif subject_match:
            flush_constraint()
            section = "constraints"
            rest = subject_match.group(2).strip()
            if rest:
                pending_constraint = rest
        elif bounds_match:
            flush_constraint()
            section = "bounds"
            rest = bounds_match.group(2).strip()
            if rest:
                bound_lines.append(rest)
        elif integer_match:
            raise ValueError("Integer, binary and semi-continuous variables are out of scope.")
        elif end_match:
            flush_constraint()
            break
        elif section == "objective":
            objective_chunks.append(line)
        elif section == "constraints":
            pending_constraint = (
                f"{pending_constraint} {line}".strip() if pending_constraint else line
            )
            if _COMPARISON_RE.search(line):
                flush_constraint()
        elif section == "bounds":
            bound_lines.append(line)
        else:
            raise ValueError(f"Content outside a recognized section: {line!r}")

    flush_constraint()

    objective_text = " ".join(objective_chunks)
    if ":" in objective_text:
        _, objective_text = objective_text.split(":", 1)
    objective_coeffs, objective_constant = _parse_linear_expression(objective_text)

    constraints = [_parse_constraint(chunk) for chunk in constraint_chunks]

    variables: list[str] = []
    variable_index: dict[str, int] = {}

    def add_variable(name: str) -> None:
        if name not in variable_index:
            variable_index[name] = len(variables)
            variables.append(name)

    for variable in objective_coeffs:
        add_variable(variable)
    for _, coefficients, _, _ in constraints:
        for variable in coefficients:
            add_variable(variable)

    bounds: dict[str, list[float]] = {}
    for line in bound_lines:
        parsed = _parse_bound_line(line)
        if parsed is None:
            continue
        variable, lower, upper = parsed
        add_variable(variable)
        if variable not in bounds:
            bounds[variable] = [0.0, float("inf")]
        if lower is not None:
            bounds[variable][0] = lower
        if upper is not None:
            bounds[variable][1] = upper

    n = len(variables)
    m = len(constraints)
    A = np.zeros((m, n), dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)
    row_sense: list[RowSense] = []
    row_names: list[str] = []

    for i, (row_name, coefficients, sense_i, rhs) in enumerate(constraints):
        for variable, value in coefficients.items():
            A[i, variable_index[variable]] = value
        b[i] = rhs
        row_sense.append(sense_i)
        row_names.append(row_name or f"r{i + 1}")

    c = np.zeros(n, dtype=np.float64)
    for variable, value in objective_coeffs.items():
        c[variable_index[variable]] = value

    lb = np.zeros(n, dtype=np.float64)
    ub = np.full(n, np.inf, dtype=np.float64)
    for j, variable in enumerate(variables):
        if variable in bounds:
            lb[j], ub[j] = bounds[variable]

    return LPProblem(
        c=c,
        A=A,
        b=b,
        row_sense=row_sense,
        lb=lb,
        ub=ub,
        sense=sense,
        name=source.stem,
        var_names=variables,
        row_names=row_names,
        constant=objective_constant,
    )


def read_mps(path: str | Path) -> LPProblem:
    """Read a linear program in free-format MPS format.

    Sections ``NAME``, ``OBJSENSE``, ``OBJNAME``, ``ROWS``, ``COLUMNS``,
    ``RHS``, ``RANGES``, ``BOUNDS`` and ``ENDATA`` are supported.  Integer,
    binary and semi-continuous variables are rejected.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()

    name = source.stem
    sense: Sense = "min"
    section: str | None = None
    row_types: dict[str, str] = {}
    row_order: list[str] = []
    columns: dict[str, dict[str, float]] = {}
    variable_order: list[str] = []
    rhs: dict[str, float] = {}
    ranges: dict[str, float] = {}
    bounds: dict[str, list[float]] = {}
    objective_name: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            upper = line.upper()
            if "SENSE" in upper and "MAX" in upper:
                sense = "max"
            continue

        tokens = line.split()
        head = tokens[0].upper()

        section_keywords = {
            "NAME",
            "OBJSENSE",
            "OBJNAME",
            "ROWS",
            "COLUMNS",
            "RHS",
            "RANGES",
            "BOUNDS",
            "ENDATA",
            "SOS",
            "QMATRIX",
            "QUADOBJ",
            "QCMATRIX",
            "INDICATORS",
        }
        is_section_header = head in section_keywords and (
            head in {"NAME", "OBJSENSE", "OBJNAME", "ENDATA"}
            or head in {"SOS", "QMATRIX", "QUADOBJ", "QCMATRIX", "INDICATORS"}
            or len(tokens) == 1
        )
        if is_section_header:
            section = head
            if head == "NAME" and len(tokens) > 1:
                name = tokens[1]
            elif head == "OBJSENSE" and len(tokens) > 1:
                if tokens[1].upper().startswith("MAX"):
                    sense = "max"
                elif tokens[1].upper().startswith("MIN"):
                    sense = "min"
            elif head == "OBJNAME" and len(tokens) > 1:
                objective_name = tokens[1]
            elif head in {"SOS", "QMATRIX", "QUADOBJ", "QCMATRIX", "INDICATORS"}:
                raise ValueError(f"{head} sections are out of scope")
            continue

        if section == "OBJSENSE":
            if head.startswith("MAX"):
                sense = "max"
            elif head.startswith("MIN"):
                sense = "min"
            continue
        if section == "OBJNAME":
            if tokens:
                objective_name = tokens[0]
            continue

        if section == "ROWS":
            if len(tokens) < 2:
                continue
            row_type = tokens[0].upper()
            row_name = tokens[1]
            if row_type not in {"N", "L", "G", "E"}:
                raise ValueError(f"Unknown row type {row_type!r}")
            if row_type == "N":
                if objective_name is None:
                    objective_name = row_name
                continue
            row_types[row_name] = row_type
            row_order.append(row_name)
        elif section == "COLUMNS":
            if len(tokens) >= 2 and tokens[1].strip("'\"").upper() == "MARKER":
                marker = tokens[2].upper() if len(tokens) > 2 else ""
                if "INTORG" in marker or "INTEND" in marker:
                    raise ValueError("Integer variables are out of scope.")
                continue
            variable = tokens[0]
            if variable not in columns:
                columns[variable] = {}
                variable_order.append(variable)
            for k in range(1, len(tokens) - 1, 2):
                row_name = tokens[k]
                value = float(tokens[k + 1])
                columns[variable][row_name] = columns[variable].get(row_name, 0.0) + value
        elif section == "RHS":
            for k in range(1, len(tokens) - 1, 2):
                row_name = tokens[k]
                rhs[row_name] = float(tokens[k + 1])
        elif section == "RANGES":
            for k in range(1, len(tokens) - 1, 2):
                row_name = tokens[k]
                ranges[row_name] = float(tokens[k + 1])
        elif section == "BOUNDS":
            if len(tokens) < 3:
                continue
            bound_type = tokens[0].upper()
            variable = tokens[2]
            value = float(tokens[3]) if len(tokens) > 3 else 0.0
            if bound_type in {"BV", "LI", "UI", "SC"}:
                raise ValueError("Integer, binary and semi-continuous variables are out of scope.")
            if variable not in columns:
                columns[variable] = {}
                variable_order.append(variable)
            if variable not in bounds:
                bounds[variable] = [0.0, float("inf")]
            if bound_type == "UP":
                bounds[variable][1] = value
            elif bound_type == "LO":
                bounds[variable][0] = value
            elif bound_type == "FX":
                bounds[variable][0] = value
                bounds[variable][1] = value
            elif bound_type == "FR":
                bounds[variable][0] = -float("inf")
                bounds[variable][1] = float("inf")
            elif bound_type == "MI":
                bounds[variable][0] = -float("inf")
            elif bound_type == "PL":
                bounds[variable][1] = float("inf")
            else:
                raise ValueError(f"Unsupported bound type {bound_type!r}")
        elif section == "ENDATA":
            break

    if objective_name is None:
        raise ValueError("MPS file has no objective row")

    rows: list[dict[str, float]] = []
    row_senses: list[RowSense] = []
    rhs_values: list[float] = []
    row_names: list[str] = []
    row_sense_map: dict[str, RowSense] = {"L": "<=", "G": ">=", "E": "="}

    for row_name in row_order:
        row_type = row_types[row_name]
        coefficients = {
            variable: columns[variable].get(row_name, 0.0)
            for variable in variable_order
            if row_name in columns[variable]
        }
        rows.append(coefficients)
        row_senses.append(row_sense_map[row_type])
        rhs_values.append(rhs.get(row_name, 0.0))
        row_names.append(row_name)

    extra_rows: list[tuple[dict[str, float], RowSense, float, str]] = []
    for index, row_name in enumerate(list(row_names)):
        if row_name not in ranges:
            continue
        range_value = ranges[row_name]
        if abs(range_value) <= 1e-15:
            continue
        base = rows[index]
        base_rhs = rhs_values[index]
        if row_senses[index] == "<=":
            extra_rows.append(
                (
                    {variable: -value for variable, value in base.items()},
                    "<=",
                    -(base_rhs - abs(range_value)),
                    f"{row_name}_lo",
                )
            )
        elif row_senses[index] == ">=":
            extra_rows.append((base, "<=", base_rhs + abs(range_value), f"{row_name}_hi"))
        elif range_value > 0.0:
            row_senses[index] = ">="
            extra_rows.append((base, "<=", base_rhs + range_value, f"{row_name}_hi"))
        else:
            row_senses[index] = "<="
            extra_rows.append((base, ">=", base_rhs + range_value, f"{row_name}_lo"))

    for coefficients, sense_i, rhs_i, row_name in extra_rows:
        rows.append(coefficients)
        row_senses.append(sense_i)
        rhs_values.append(rhs_i)
        row_names.append(row_name)

    variable_index = {variable: j for j, variable in enumerate(variable_order)}
    n = len(variable_order)
    m = len(rows)
    A = np.zeros((m, n), dtype=np.float64)
    b = np.asarray(rhs_values, dtype=np.float64)

    for i, coefficients in enumerate(rows):
        for variable, value in coefficients.items():
            A[i, variable_index[variable]] = value

    c = np.zeros(n, dtype=np.float64)
    for variable in variable_order:
        c[variable_index[variable]] = columns[variable].get(objective_name, 0.0)

    lb = np.zeros(n, dtype=np.float64)
    ub = np.full(n, np.inf, dtype=np.float64)
    for j, variable in enumerate(variable_order):
        if variable in bounds:
            lb[j], ub[j] = bounds[variable]

    return LPProblem(
        c=c,
        A=A,
        b=b,
        row_sense=row_senses,
        lb=lb,
        ub=ub,
        sense=sense,
        name=name,
        var_names=variable_order,
        row_names=row_names,
        constant=-rhs.get(objective_name, 0.0),
    )


def _parse_linear_expression(text: str) -> tuple[dict[str, float], float]:
    tokens = _tokenize(text)
    coefficients: dict[str, float] = {}
    constant = 0.0
    sign = 1.0
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token == "+":
            sign = 1.0
            i += 1
        elif token == "-":
            sign = -1.0
            i += 1
        elif token == "*":
            i += 1
        elif _is_number(token):
            value = float(token)
            j = i + 1
            if j < len(tokens) and tokens[j] == "*":
                j += 1
            if j < len(tokens) and _is_identifier(tokens[j]):
                variable = tokens[j]
                coefficients[variable] = coefficients.get(variable, 0.0) + sign * value
                sign = 1.0
                i = j + 1
            else:
                constant += sign * value
                sign = 1.0
                i += 1
        elif _is_identifier(token):
            coefficients[token] = coefficients.get(token, 0.0) + sign
            sign = 1.0
            i += 1
        else:
            raise ValueError(f"Unexpected token {token!r} in expression {text!r}")

    return coefficients, constant


def _parse_constraint(text: str) -> tuple[str, dict[str, float], RowSense, float]:
    match = _COMPARISON_RE.search(text)
    if match is None:
        raise ValueError(f"Constraint has no comparison operator: {text!r}")

    left_text = text[: match.start()].strip()
    right_text = text[match.end() :].strip()
    operator = match.group(1)
    if operator == "<=":
        sense: RowSense = "<="
    elif operator == ">=":
        sense = ">="
    else:
        sense = "="

    row_name = ""
    if ":" in left_text:
        label, left_text = left_text.split(":", 1)
        row_name = label.strip()

    left_coefficients, left_constant = _parse_linear_expression(left_text)
    right_coefficients, right_constant = _parse_linear_expression(right_text)

    coefficients: dict[str, float] = {}
    for variable in set(left_coefficients) | set(right_coefficients):
        value = left_coefficients.get(variable, 0.0) - right_coefficients.get(variable, 0.0)
        if value != 0.0:
            coefficients[variable] = value

    return row_name, coefficients, sense, right_constant - left_constant


def _parse_bound_line(text: str) -> tuple[str, float | None, float | None] | None:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.endswith(" free"):
        variable = stripped[:-4].strip()
        if not _is_identifier(variable):
            raise ValueError(f"Invalid free-variable declaration: {text!r}")
        return variable, -float("inf"), float("inf")

    parts = [part.strip() for part in re.split(r"(<=|>=|=)", stripped)]
    if len(parts) == 3:
        left, operator, right = parts
        left_kind, left_value = _operand(left)
        right_kind, right_value = _operand(right)
        if left_kind == "variable" and right_kind == "number":
            variable = str(left_value)
            value = float(right_value)
            if operator == "<=":
                return variable, None, value
            if operator == ">=":
                return variable, value, None
            return variable, value, value
        if left_kind == "number" and right_kind == "variable":
            variable = str(right_value)
            value = float(left_value)
            if operator == "<=":
                return variable, value, None
            if operator == ">=":
                return variable, None, value
            return variable, value, value
        raise ValueError(f"Invalid bound line: {text!r}")

    if len(parts) == 5:
        left, first_operator, middle, second_operator, right = parts
        left_kind, left_value = _operand(left)
        middle_kind, middle_value = _operand(middle)
        right_kind, right_value = _operand(right)
        if left_kind != "number" or middle_kind != "variable" or right_kind != "number":
            raise ValueError(f"Invalid two-sided bound: {text!r}")
        variable = str(middle_value)
        first = float(left_value)
        second = float(right_value)
        if first_operator == second_operator == "<=":
            return variable, first, second
        if first_operator == second_operator == ">=":
            return variable, second, first
        raise ValueError(f"Mixed comparison directions in bound: {text!r}")

    raise ValueError(f"Cannot parse bound line: {text!r}")


def _operand(text: str) -> tuple[str, str | float]:
    normalized = text.strip().lower()
    if normalized in {"inf", "+inf", "infinity", "+infinity"}:
        return "number", float("inf")
    if normalized in {"-inf", "-infinity"}:
        return "number", -float("inf")

    coefficients, constant = _parse_linear_expression(text)
    if len(coefficients) == 1:
        variable, coefficient = next(iter(coefficients.items()))
        if abs(coefficient - 1.0) <= 1e-12 and abs(constant) <= 1e-12:
            return "variable", variable
    if not coefficients:
        return "number", constant
    raise ValueError(f"Cannot interpret bound operand: {text!r}")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    for match in _TOKEN_RE.finditer(text):
        if text[position : match.start()].strip():
            raise ValueError(f"Unexpected text {text[position:match.start()]!r}")
        group = match.lastgroup
        if group is None:
            raise ValueError(f"Token without a group in {text!r}")
        tokens.append(match.group(group))
        position = match.end()
    if text[position:].strip():
        raise ValueError(f"Unexpected text {text[position:]!r}")
    return tokens


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_identifier(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", token))
