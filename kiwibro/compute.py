"""Offline deterministic compute: exact math, unit conversion, date math and table analysis.

Small local models are unreliable at arithmetic, unit conversion and date
arithmetic, so these tools return exact answers. Nothing here calls eval/exec:
math is parsed with ``ast`` and checked against an allow-list before a small
recursive evaluator walks it. Tables are streamed row by row, never loaded whole.
"""

from __future__ import annotations

import ast
import calendar
import csv
import heapq
import json
import math
import operator
import random
import re
import statistics
from array import array
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

from .extensions import ExtensionTool, schema, tool_result

MAX_EXPRESSION_CHARS = 500
MAX_POWER_EXPONENT = 100_000
MAX_POWER_DIGITS = 20_000
MAX_FACTORIAL = 3000
DETAIL_DIGITS = 40

_THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", re.IGNORECASE)
_PERCENT_CHANGE = re.compile(r"(\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)\s*%(?!\s*(?:of\b|[\d(]))", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])")
_BETWEEN_DIGITS_X = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)", re.IGNORECASE)
_LEADING = re.compile(r"^\s*(?:what\s+is|what's|calculate|compute)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"\s*[?=]\s*$")


def _preprocess_math(text: str) -> str:
    expr = text.replace("×", "*").replace("·", "*").replace("−", "-").replace("÷", "/").replace("^", "**")
    expr = _LEADING.sub("", expr)
    expr = _TRAILING.sub("", expr).strip()
    expr = _PERCENT_OF.sub(lambda m: f"({m.group(1)}/100)*", expr)
    expr = _PERCENT_CHANGE.sub(lambda m: f"({m.group(1)}*(1{m.group(2)}{m.group(3)}/100))", expr)
    expr = _PERCENT.sub(lambda m: f"({m.group(1)}/100)", expr)
    expr = _BETWEEN_DIGITS_X.sub("*", expr)
    if "(" not in expr:
        expr = _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), expr)
    return expr


def _num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError("expected a number")
    return float(value)


def _checked(fn: Callable[..., float], *args: float) -> float:
    try:
        return fn(*args)
    except ValueError as exc:
        raise ValueError(f"math domain error: {exc}") from exc
    except OverflowError as exc:
        raise ValueError("result is too large") from exc


def _ints(values: list[Any]) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("gcd and lcm take whole numbers")
        out.append(value)
    return out


def _to_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number")
    return value


def _mean(*values: Any) -> float:
    if not values:
        raise ValueError("mean needs at least one value")
    return statistics.fmean([_num(v) for v in values])


def _median(*values: Any) -> float:
    if not values:
        raise ValueError("median needs at least one value")
    return statistics.median([_num(v) for v in values])


def _factorial(n: Any) -> int:
    n = _to_int(n, "factorial")
    if n < 0 or n > MAX_FACTORIAL:
        raise ValueError(f"factorial takes 0 to {MAX_FACTORIAL}")
    return math.factorial(n)


def _log(x: Any, base: Any = None) -> float:
    if base is None:
        return _checked(math.log, _num(x))
    return _checked(math.log, _num(x), _num(base))


def _round(x: Any, digits: Any = None) -> Any:
    if digits is None:
        return round(x)
    return round(x, _to_int(digits, "round digits"))


def _power(base: Any, exp: Any) -> Any:
    """Exact int**int with size guards; everything else goes through float."""
    if isinstance(base, int) and isinstance(exp, int):
        if exp < 0:
            if base == 0:
                raise ValueError("division by zero")
            return Fraction(1, base ** -exp)
        if exp > MAX_POWER_EXPONENT or (abs(base) > 1 and exp * math.log10(abs(base)) > MAX_POWER_DIGITS):
            raise ValueError("result is too large")
        return base**exp
    try:
        result = _num(base) ** _num(exp)
    except OverflowError as exc:
        raise ValueError("result is too large") from exc
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc
    if isinstance(result, complex):
        raise ValueError("math domain error: negative base with a fractional exponent")
    return result


def _sqrt(x: Any) -> float:
    return _checked(math.sqrt, _num(x))


def _cbrt(x: Any) -> float:
    value = _num(x)
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def _unary_float(fn: Callable[[float], float]) -> Callable[[Any], float]:
    return lambda x: _checked(fn, _num(x))


def _binary_float(fn: Callable[[float, float], float]) -> Callable[[Any, Any], float]:
    return lambda a, b: _checked(fn, _num(a), _num(b))


def _gcd(*values: Any) -> int:
    return math.gcd(*_ints(list(values)))


def _lcm(*values: Any) -> int:
    result = 1
    for value in _ints(list(values)):
        if value == 0:
            return 0
        result = abs(result * value) // math.gcd(result, value)
    return result


def _comb(n: Any, k: Any) -> int:
    return math.comb(_to_int(n, "comb n"), _to_int(k, "comb k"))


def _perm(n: Any, k: Any = None) -> int:
    if k is None:
        return math.perm(_to_int(n, "perm n"))
    return math.perm(_to_int(n, "perm n"), _to_int(k, "perm k"))


def _hypot(*values: Any) -> float:
    return _checked(math.hypot, *[_num(v) for v in values])


def _min(*values: Any) -> Any:
    if not values:
        raise ValueError("min needs at least one value")
    return min(values)


def _max(*values: Any) -> Any:
    if not values:
        raise ValueError("max needs at least one value")
    return max(values)


def _sum(*values: Any) -> Any:
    return sum(values)


def _floor(x: Any) -> int:
    return math.floor(_num(x))


def _ceil(x: Any) -> int:
    return math.ceil(_num(x))


def _trunc(x: Any) -> int:
    return math.trunc(_num(x))


MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": _sqrt,
    "cbrt": _cbrt,
    "exp": _unary_float(math.exp),
    "ln": _unary_float(math.log),
    "log": _log,
    "log10": _unary_float(math.log10),
    "log2": _unary_float(math.log2),
    "sin": _unary_float(math.sin),
    "cos": _unary_float(math.cos),
    "tan": _unary_float(math.tan),
    "asin": _unary_float(math.asin),
    "acos": _unary_float(math.acos),
    "atan": _unary_float(math.atan),
    "atan2": _binary_float(math.atan2),
    "sinh": _unary_float(math.sinh),
    "cosh": _unary_float(math.cosh),
    "tanh": _unary_float(math.tanh),
    "degrees": _unary_float(math.degrees),
    "radians": _unary_float(math.radians),
    "abs": abs,
    "round": _round,
    "floor": _floor,
    "ceil": _ceil,
    "trunc": _trunc,
    "min": _min,
    "max": _max,
    "sum": _sum,
    "mean": _mean,
    "median": _median,
    "factorial": _factorial,
    "gcd": _gcd,
    "lcm": _lcm,
    "comb": _comb,
    "perm": _perm,
    "hypot": _hypot,
    "pow": _power,
}
MATH_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}
MAX_PLAIN_DIGITS = 4000
MAX_INT_BITS = 200_000


def _exact_div(a: Any, b: Any) -> Any:
    if isinstance(a, int) and isinstance(b, int) and b and a % b == 0:
        return a // b
    return a / b


def _guarded(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def run(a: Any, b: Any) -> Any:
        try:
            result = fn(a, b)
        except ZeroDivisionError as exc:
            raise ValueError("division by zero") from exc
        except OverflowError as exc:
            raise ValueError("result is too large") from exc
        if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
            raise ValueError("result is too large")
        return result

    return run


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: _guarded(operator.add),
    ast.Sub: _guarded(operator.sub),
    ast.Mult: _guarded(operator.mul),
    ast.Div: _guarded(_exact_div),
    ast.FloorDiv: _guarded(operator.floordiv),
    ast.Mod: _guarded(operator.mod),
    ast.Pow: _guarded(_power),
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_call(node: ast.Call) -> Any:
    if not isinstance(node.func, ast.Name) or node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
        raise ValueError("unsupported syntax")
    name = node.func.id
    fn = MATH_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"unknown function '{name}'")
    args = [_eval_node(arg) for arg in node.args]
    try:
        return fn(*args)
    except TypeError as exc:
        raise ValueError(f"wrong number or type of arguments for {name}()") from exc
    except OverflowError as exc:
        raise ValueError("result is too large") from exc


def _eval_node(node: ast.AST) -> Any:
    """Walk the allow-listed AST. Anything not listed is rejected, never executed."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in MATH_CONSTANTS:
            raise ValueError(f"unknown name '{node.id}'; use numbers, pi, e or tau")
        return MATH_CONSTANTS[node.id]
    if isinstance(node, ast.Call):
        return _eval_call(node)
    raise ValueError("unsupported syntax")


def _preprocess_or_fail(text: str) -> str:
    expr = _preprocess_math(text)
    if not expr:
        raise ValueError("nothing to calculate; give an expression such as '15% of 240'")
    return expr


def eval_math(text: str) -> Any:
    expr = _preprocess_or_fail(text)
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("could not parse that as math; check operators and parentheses") from exc
    return _eval_node(tree)


def _digit_count(value: int) -> int:
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    digits = int(math.log10(magnitude)) + 1
    if magnitude >= 10**digits:
        digits += 1
    elif magnitude < 10 ** (digits - 1):
        digits -= 1
    return digits


def _trim(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def _scientific_int(value: int, digits: int) -> str:
    magnitude = abs(value)
    lead = magnitude // 10 ** (digits - 8) if digits > 8 else magnitude * 10 ** (8 - digits)
    text = str(lead)
    sign = "-" if value < 0 else ""
    return f"{sign}{_trim(text[0] + '.' + text[1:])}e+{digits - 1}"


def _format_int(value: int) -> tuple[str, str]:
    digits = _digit_count(value)
    if digits > MAX_PLAIN_DIGITS:
        return _scientific_int(value, digits), f"exact value has {digits} digits"
    detail = f"≈ {_scientific_int(value, digits)} ({digits} digits)" if digits > DETAIL_DIGITS else ""
    return str(value), detail


def _format_float(value: float) -> str:
    if math.isnan(value):
        raise ValueError("math domain error: result is not a number")
    if math.isinf(value):
        raise ValueError("result is too large")
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    magnitude = abs(value)
    if 1e-6 <= magnitude < 1e15:
        decimals = max(0, 11 - math.floor(math.log10(magnitude)))
        return _trim(f"{value:.{decimals}f}")
    mantissa, exponent = f"{value:.11e}".split("e")
    return f"{_trim(mantissa)}e{exponent}"


def format_math(value: Any) -> tuple[str, str]:
    if isinstance(value, Fraction):
        try:
            value = value.numerator if value.denominator == 1 else float(value)
        except OverflowError as exc:
            raise ValueError("result is too large") from exc
    if isinstance(value, int):
        return _format_int(value)
    return _format_float(float(value)), ""


# --- units -------------------------------------------------------------------

_CATEGORIES: dict[str, dict[str, str]] = {
    "length": {
        "m": "1", "km": "1000", "cm": "1/100", "mm": "1/1000", "um": "1/1000000", "nm": "1/1000000000",
        "mi": "1609.344", "yd": "0.9144", "ft": "0.3048", "in": "0.0254", "nmi": "1852",
        "au": "149597870700", "ly": "9460730472580800",
    },
    "mass": {
        "kg": "1", "g": "1/1000", "mg": "1/1000000", "ug": "1/1000000000", "t": "1000", "lb": "0.45359237",
        "oz": "0.028349523125", "st": "6.35029318", "short ton": "907.18474", "long ton": "1016.0469088",
    },
    "volume": {
        "l": "1", "ml": "1/1000", "cl": "1/100", "dl": "1/10", "m3": "1000", "cm3": "1/1000",
        "gal": "3.785411784", "imperial gallon": "4.54609", "qt": "0.946352946", "pt": "0.473176473",
        "cup": "0.2365882365", "floz": "0.0295735295625", "tbsp": "0.01478676478125", "tsp": "0.00492892159375",
    },
    "time": {
        "s": "1", "ms": "1/1000", "us": "1/1000000", "ns": "1/1000000000", "min": "60", "h": "3600",
        "day": "86400", "week": "604800", "month": "2629746", "year": "31557600",
        "decade": "315576000", "century": "3155760000",
    },
    "data": {
        "bit": "1/8", "B": "1", "kbit": "125", "Mbit": "125000", "Gbit": "125000000",
        "KB": "1000", "MB": "1000000", "GB": "1000000000", "TB": "1000000000000", "PB": "1000000000000000",
        "KiB": "1024", "MiB": "1048576", "GiB": "1073741824", "TiB": "1099511627776", "PiB": "1125899906842624",
    },
    "speed": {"m/s": "1", "km/h": "5/18", "mph": "0.44704", "knot": "1852/3600", "ft/s": "0.3048"},
    "area": {
        "m2": "1", "km2": "1000000", "cm2": "1/10000", "mm2": "1/1000000", "ha": "10000", "acre": "4046.8564224",
        "ft2": "0.09290304", "in2": "0.00064516", "yd2": "0.83612736", "mi2": "2589988.110336",
    },
    "energy": {
        "J": "1", "kJ": "1000", "MJ": "1000000", "cal": "4.184", "kcal": "4184", "Wh": "3600", "kWh": "3600000",
        "eV": "1.602176634e-19", "BTU": "1055.05585262",
    },
    "pressure": {
        "Pa": "1", "hPa": "100", "kPa": "1000", "MPa": "1000000", "bar": "100000", "mbar": "100",
        "atm": "101325", "psi": "6894.757293168", "mmHg": "133.322387415", "inHg": "3386.389",
    },
    "power": {"W": "1", "kW": "1000", "MW": "1000000", "hp": "745.69987158227", "PS": "735.49875"},
    "temperature": {"C": "0", "F": "0", "K": "0"},
}
_CATEGORY_OF: dict[str, str] = {key: cat for cat, units in _CATEGORIES.items() for key in units}
_FACTOR_OF: dict[str, Fraction] = {key: Fraction(f) for units in _CATEGORIES.values() for key, f in units.items()}
_CASE_ALIASES: dict[str, str] = {"kB": "KB", "Kb": "kbit", "Mb": "Mbit", "Gb": "Gbit"}
_WORDS: dict[str, str] = {
    "meter": "m", "metre": "m", "kilometre": "km", "centimeter": "cm", "centimetre": "cm",
    "millimeter": "mm", "millimetre": "mm", "micrometer": "um", "micron": "um", "nanometer": "nm",
    "mile": "mi", "yard": "yd", "foot": "ft", "feet": "ft", "inch": "in", "inches": "in",
    "nautical mile": "nmi", "astronomical unit": "au", "light year": "ly", "lightyear": "ly", "light-year": "ly",
    "kilogram": "kg", "gram": "g", "gramme": "g", "milligram": "mg", "microgram": "ug", "tonne": "t",
    "ton": "t", "metric ton": "t", "pound": "lb", "lbs": "lb", "ounce": "oz", "stone": "st",
    "liter": "l", "litre": "l", "milliliter": "ml", "millilitre": "ml", "centiliter": "cl", "deciliter": "dl",
    "cubic meter": "m3", "cubic metre": "m3", "cubic centimeter": "cm3", "cubic centimetre": "cm3", "cc": "cm3",
    "gallon": "gal", "us gallon": "gal", "imperial gallon": "imperial gallon", "quart": "qt", "pint": "pt",
    "cup": "cup", "fl oz": "floz", "fl. oz": "floz", "fluid ounce": "floz", "tablespoon": "tbsp", "tbs": "tbsp",
    "teaspoon": "tsp",
    "sec": "s", "second": "s", "millisecond": "ms", "microsecond": "us", "nanosecond": "ns", "minute": "min",
    "hour": "h", "hr": "h", "d": "day", "wk": "week", "yr": "year", "centuries": "century",
    "byte": "B", "kilobyte": "KB", "megabyte": "MB", "gigabyte": "GB", "terabyte": "TB", "petabyte": "PB",
    "kibibyte": "KiB", "mebibyte": "MiB", "gibibyte": "GiB", "tebibyte": "TiB", "pebibyte": "PiB",
    "bit": "bit", "b": "bit", "kilobit": "kbit", "megabit": "Mbit", "gigabit": "Gbit",
    "kb": "KB", "mb": "MB", "gb": "GB",
    "kph": "km/h", "kmh": "km/h", "kmph": "km/h", "kilometer per hour": "km/h", "kilometers per hour": "km/h",
    "kilometres per hour": "km/h", "km per hour": "km/h", "mile per hour": "mph", "miles per hour": "mph",
    "mi/h": "mph", "meter per second": "m/s", "meters per second": "m/s", "metres per second": "m/s", "mps": "m/s",
    "kn": "knot", "kt": "knot", "feet per second": "ft/s", "foot per second": "ft/s", "fps": "ft/s",
    "hectare": "ha", "joule": "J", "kilojoule": "kJ", "calorie": "cal", "kilocalorie": "kcal",
    "watt hour": "Wh", "kilowatt hour": "kWh", "electronvolt": "eV",
    "pascal": "Pa", "hectopascal": "hPa", "kilopascal": "kPa", "megapascal": "MPa", "atmosphere": "atm",
    "torr": "mmHg", "mm hg": "mmHg", "in hg": "inHg",
    "watt": "W", "kilowatt": "kW", "megawatt": "MW", "horsepower": "hp",
    "celsius": "C", "centigrade": "C", "fahrenheit": "F", "kelvin": "K",
}
_ALIASES: dict[str, str] = dict(_WORDS)
for _key in _CATEGORY_OF:
    _ALIASES.setdefault(_key.lower(), _key)
_UNIT_EXAMPLES = "km, mi, kg, lb, L, gal, MB, MiB, km/h, °F"
_KELVIN_OFFSET = Fraction("273.15")
_UNIT_LINE = re.compile(
    r"^\s*(?P<num>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?)\s*(?P<src>(?:[^\W\d]|°)[^\n]*?)"
    r"(?:\s+(?:to|in|into|as)\s+|\s*(?:->|=>|→)\s*)(?P<dst>[^\n]*?)\s*\??\s*$",
    re.IGNORECASE,
)


def _clean_unit(raw: str) -> str:
    text = raw.replace("°", "").replace("²", "2").replace("³", "3").replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\b(?:degrees?|deg)\b", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _resolve_unit(raw: str) -> str | None:
    text = _clean_unit(raw)
    if not text:
        return None
    if text in _CATEGORY_OF:
        return text
    if text in _CASE_ALIASES:
        return _CASE_ALIASES[text]
    lower = text.lower()
    if lower in _ALIASES:
        return _ALIASES[lower]
    squared = re.fullmatch(r"(sq|square|cu|cubic)\.? (.+)", lower)
    if squared:
        base = _resolve_unit(squared.group(2))
        power = "2" if squared.group(1) in ("sq", "square") else "3"
        return base + power if base and base + power in _CATEGORY_OF else None
    if lower.endswith("s") and lower[:-1] in _ALIASES:
        return _ALIASES[lower[:-1]]
    return None


def _require_unit(raw: str, kind: str | None = None) -> str:
    key = _resolve_unit(raw)
    if key is None:
        examples = ", ".join(list(_CATEGORIES[kind])[:6]) if kind else _UNIT_EXAMPLES
        raise ValueError(f"unknown unit '{raw.strip()}'; try units like {examples}")
    return key


def _to_kelvin(value: Fraction, key: str) -> Fraction:
    if key == "C":
        return value + _KELVIN_OFFSET
    if key == "F":
        return (value - 32) * Fraction(5, 9) + _KELVIN_OFFSET
    return value


def _from_kelvin(value: Fraction, key: str) -> Fraction:
    if key == "C":
        return value - _KELVIN_OFFSET
    if key == "F":
        return (value - _KELVIN_OFFSET) * Fraction(9, 5) + 32
    return value


def unit_result(text: str) -> tuple[str, str] | None:
    match = _UNIT_LINE.match(text)
    if not match:
        return None
    number = Fraction(match["num"].replace(",", ""))
    src = _require_unit(match["src"])
    dst = _require_unit(match["dst"], _CATEGORY_OF[src])
    category = _CATEGORY_OF[src]
    if category != _CATEGORY_OF[dst]:
        raise ValueError(f"cannot convert {category} to {_CATEGORY_OF[dst]}")
    if category == "temperature":
        value = _from_kelvin(_to_kelvin(number, src), dst)
        return f"{format_math(value)[0]} °{dst}", ""
    ratio = _FACTOR_OF[src] / _FACTOR_OF[dst]
    value = number * ratio
    return f"{format_math(value)[0]} {dst}", f"1 {src} = {format_math(ratio)[0]} {dst}"


# --- dates -------------------------------------------------------------------

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_UNIT_WORD = r"(days?|weeks?|months?|years?)"
_KEYWORD_DAYS = {"today": 0, "now": 0, "tomorrow": 1, "yesterday": -1}
_ISO_DATE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_DOTTED_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
_MONTH_FIRST = re.compile(rf"{_MONTH}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", re.IGNORECASE)
_DAY_FIRST = re.compile(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+(\d{{4}})", re.IGNORECASE)
_DIFFERENCE = re.compile(r"(?:days?\s+(?:between|from)|difference\s+between)\s+(.+?)\s+(?:and|to)\s+(.+)", re.IGNORECASE)
_WEEKDAY_OF = re.compile(
    r"(?:weekday\s+of|day\s+of\s+(?:the\s+)?week(?:\s+of)?|what\s+day\s+(?:is|was)|which\s+day\s+(?:is|was))\s+(.+)",
    re.IGNORECASE,
)
_AGO = re.compile(rf"(\d+)\s*{_UNIT_WORD}\s+ago", re.IGNORECASE)
_IN_N = re.compile(rf"in\s+(\d+)\s*{_UNIT_WORD}", re.IGNORECASE)
_N_FROM = re.compile(rf"(\d+)\s*{_UNIT_WORD}\s+(after|before|from)\s+(.+)", re.IGNORECASE)
_SHIFT = re.compile(rf"(.+?)\s*([+-])\s*(\d+)\s*{_UNIT_WORD}", re.IGNORECASE)


def _date_parts(text: str) -> tuple[int, int, int] | None:
    if m := _ISO_DATE.fullmatch(text):
        year, month, day = m.groups()
    elif m := _DOTTED_DATE.fullmatch(text):
        day, month, year = m.groups()
    elif m := _MONTH_FIRST.fullmatch(text):
        month_word, day, year = m.groups()
        month = _MONTH_NUMBERS[month_word[:3].lower()]
    elif m := _DAY_FIRST.fullmatch(text):
        day, month_word, year = m.groups()
        month = _MONTH_NUMBERS[month_word[:3].lower()]
    else:
        return None
    return int(year), int(month), int(day)


def _try_date(text: str, today: date) -> date | None:
    """Parse one date phrase; None when the text is not a date at all."""
    cleaned = " ".join(text.strip().rstrip("?").split())
    if cleaned.lower() in _KEYWORD_DAYS:
        return today + timedelta(days=_KEYWORD_DAYS[cleaned.lower()])
    parts = _date_parts(cleaned)
    if parts is None:
        return None
    try:
        return date(*parts)
    except ValueError as exc:
        raise ValueError(f"invalid date '{cleaned}'; use a form like 2026-10-22") from exc


def _add_months(base: date, months: int) -> date:
    year, month_index = divmod(base.year * 12 + (base.month - 1) + months, 12)
    last_day = calendar.monthrange(year, month_index + 1)[1]
    return date(year, month_index + 1, min(base.day, last_day))


def _shift(base: date, amount: int, unit: str) -> date:
    kind = unit.lower().rstrip("s")
    try:
        if kind == "day":
            return base + timedelta(days=amount)
        if kind == "week":
            return base + timedelta(days=7 * amount)
        return _add_months(base, amount if kind == "month" else 12 * amount)
    except (OverflowError, ValueError) as exc:
        raise ValueError("that date is out of range") from exc


def _calendar_gap(first: date, last: date) -> tuple[int, int, int]:
    months = (last.year - first.year) * 12 + last.month - first.month
    if last.day < first.day:
        months -= 1
    years, rest = divmod(months, 12)
    return years, rest, (last - _add_months(first, months)).days


def _describe_date(day: date) -> tuple[str, str]:
    return f"{day.isoformat()} ({WEEKDAYS[day.weekday()]})", ""


def date_result(text: str, today: date) -> tuple[str, str] | None:
    if m := _DIFFERENCE.fullmatch(text):
        start, end = _try_date(m[1], today), _try_date(m[2], today)
        if start is None or end is None:
            return None
        delta = (end - start).days
        years, months, days = _calendar_gap(*sorted((start, end)))
        weeks, rest = divmod(abs(delta), 7)
        return f"{delta} days", f"{weeks} weeks {rest} days; {years} years {months} months {days} days"
    if m := _WEEKDAY_OF.fullmatch(text):
        day = _try_date(m[1], today)
        return None if day is None else _describe_date(day)
    if m := _AGO.fullmatch(text):
        return _describe_date(_shift(today, -int(m[1]), m[2]))
    if m := _IN_N.fullmatch(text):
        return _describe_date(_shift(today, int(m[1]), m[2]))
    if m := _N_FROM.fullmatch(text):
        base = _try_date(m[4], today)
        if base is None:
            return None
        sign = -1 if m[3].lower() == "before" else 1
        return _describe_date(_shift(base, sign * int(m[1]), m[2]))
    if m := _SHIFT.fullmatch(text):
        base = _try_date(m[1], today)
        if base is None:
            return None
        sign = -1 if m[2] == "-" else 1
        return _describe_date(_shift(base, sign * int(m[3]), m[4]))
    day = _try_date(text, today)
    return None if day is None else _describe_date(day)


# --- entry point -------------------------------------------------------------


def _record(kind: str, text: str, outcome: tuple[str, str]) -> dict[str, str]:
    return {"kind": kind, "input": text, "result": outcome[0], "detail": outcome[1]}


def evaluate(expression: str, *, today: date | None = None) -> dict[str, str]:
    """Evaluate one expression: date first, then units, then math."""
    text = " ".join((expression or "").split())
    if not text:
        raise ValueError("expression is empty; try '15% of 240' or '72 F to C'")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is too long (limit {MAX_EXPRESSION_CHARS} characters)")
    outcome = date_result(text, today or date.today())
    if outcome is not None:
        return _record("date", text, outcome)
    outcome = unit_result(text)
    if outcome is not None:
        return _record("units", text, outcome)
    return _record("math", text, format_math(eval_math(text)))


# --- tables ------------------------------------------------------------------

MAX_TABLE_BYTES = 500 * 1024 * 1024
MAX_JSON_BYTES = 50 * 1024 * 1024
MAX_GROUPS = 5000
MAX_DISTINCT = 10_000
MEDIAN_EXACT = 200_000
MEDIAN_GROUP_SAMPLE = 2000
CELL_CHARS = 60
MAX_LIMIT = 200
SNIFF_BYTES = 64 * 1024
AGGREGATES = ("count", "sum", "mean", "min", "max", "median")
_COMPARE_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_QUOTED_COLUMN = re.compile(r"`([^`]+)`")
_CONTAINS = re.compile(
    r"(?P<col>[A-Za-z_]\w*)\s+(?P<neg>not\s+)?contains\s+(?P<val>\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')",
    re.IGNORECASE,
)
_INTEGER_TEXT = re.compile(r"[-+]?\d+")


def _table_kind(path: Path) -> str:
    kinds = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".txt": "txt", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl"}
    kind = kinds.get(path.suffix.lower())
    if kind is None:
        raise ValueError("unsupported table type; use .csv, .tsv, .txt, .json, .jsonl or .ndjson")
    return kind


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _column_names(header: list[str]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(header, start=1):
        base = raw.strip() or f"column_{index}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return names


_DECIMAL = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _number(value: Any) -> Fraction | None:
    """Exact value of a plain decimal cell; '1/2', '2026/10', 'nan' and '1,234' stay text."""
    if isinstance(value, (int, float)):
        return Fraction(value)
    text = str(value).strip()
    if not _DECIMAL.fullmatch(text):
        return None
    return Fraction(text)


def _resolve_column(name: str, columns: list[str]) -> str:
    if name in columns:
        return name
    matches = [column for column in columns if column.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown column '{name}'; columns are: {', '.join(columns)}")


class _Table:
    """Streams the rows of one table file; only JSON arrays (capped at 50 MB) are held whole."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kind = _table_kind(path)
        self._delimiter = "\t" if self.kind == "tsv" else ","
        self._json_items: list[dict[str, Any]] = []
        self.columns = self._load_columns()

    def _open(self):
        return open(self.path, newline="", encoding="utf-8-sig", errors="replace")

    def _load_columns(self) -> list[str]:
        if self.kind == "json":
            return self._load_json()
        if self.kind == "jsonl":
            return self._scan_jsonl()
        if self.kind == "txt":
            self._delimiter = self._sniff_delimiter()
        with self._open() as handle:
            header = next(csv.reader(handle, delimiter=self._delimiter), None)
        if not header:
            raise ValueError("the table is empty")
        return _column_names(header)

    def _sniff_delimiter(self) -> str:
        with self._open() as handle:
            sample = handle.read(SNIFF_BYTES)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            return ","

    def _load_json(self) -> list[str]:
        if self.path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("this JSON file is over 50 MB; convert it to JSON Lines (.jsonl), which streams")
        with open(self.path, encoding="utf-8-sig", errors="replace") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("a JSON table must be an array of objects; use .jsonl for one object per line")
        self._json_items = data
        keys: dict[str, None] = {}
        for item in data:
            for key in item:
                keys.setdefault(str(key), None)
        return list(keys)

    def _scan_jsonl(self) -> list[str]:
        keys: dict[str, None] = {}
        with open(self.path, encoding="utf-8-sig", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = self._parse_jsonl_line(line, line_no)
                for key in item:
                    keys.setdefault(str(key), None)
        return list(keys)

    @staticmethod
    def _parse_jsonl_line(line: str, line_no: int) -> dict[str, Any]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no} is not valid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no} is not a JSON object")
        return item

    def rows(self) -> Iterator[dict[str, str]]:
        if self.kind == "json":
            for item in self._json_items:
                yield {column: _cell(item.get(column)) for column in self.columns}
            return
        if self.kind == "jsonl":
            with open(self.path, encoding="utf-8-sig", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if line.strip():
                        item = self._parse_jsonl_line(line, line_no)
                        yield {column: _cell(item.get(column)) for column in self.columns}
            return
        width = len(self.columns)
        with self._open() as handle:
            reader = csv.reader(handle, delimiter=self._delimiter)
            next(reader, None)
            for row in reader:
                padded = row[:width] + [""] * (width - len(row))
                yield dict(zip(self.columns, padded))


def _quote_columns(text: str) -> tuple[str, dict[str, str]]:
    """Swap `backtick quoted` column names for safe identifiers."""
    names: dict[str, str] = {}

    def substitute(match: re.Match[str]) -> str:
        placeholder = f"__kbcol{len(names)}__"
        names[placeholder] = match[1]
        return placeholder

    return _QUOTED_COLUMN.sub(substitute, text), names


def _constant(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        return -node.operand.value
    raise ValueError('compare columns with plain numbers or quoted text, e.g. price > 100 or country == "TR"')


def _compare_cell(op: Callable[[Any, Any], bool], cell: str, constant: Any) -> bool:
    left, right = _number(cell), _number(constant)
    if left is None or right is None:
        left, right = cell, str(constant)
    return op(left, right)


class _Where:
    """Compiles a row filter into closures. Only comparisons, and/or/not, in and contains are allowed."""

    def __init__(self, text: str, columns: list[str]) -> None:
        self._columns = columns
        expression, self._quoted = _quote_columns(text)
        expression = _CONTAINS.sub(
            lambda m: f"{'not ' if m['neg'] else ''}contains({m['col']}, {m['val']})",
            expression,
        )
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError('could not read the where clause; try: price > 100 and country == "TR"') from exc
        self._predicate = self._build(tree.body)

    def matches(self, row: dict[str, str]) -> bool:
        return self._predicate(row)

    def _column(self, node: ast.AST) -> str:
        if not isinstance(node, ast.Name):
            raise ValueError("start each comparison with a column name, e.g. price > 100")
        return _resolve_column(self._quoted.get(node.id, node.id), self._columns)

    def _build(self, node: ast.AST) -> Callable[[dict[str, str]], bool]:
        if isinstance(node, ast.BoolOp):
            parts = [self._build(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return lambda row: all(part(row) for part in parts)
            return lambda row: any(part(row) for part in parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self._build(node.operand)
            return lambda row: not inner(row)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "contains":
            return self._contains(node)
        raise ValueError(
            "unsupported where clause; use comparisons such as price > 100, country == \"TR\", "
            "x in [1, 2] or name contains \"text\", joined with and / or / not"
        )

    def _compare(self, node: ast.Compare) -> Callable[[dict[str, str]], bool]:
        if len(node.ops) != 1:
            raise ValueError("write one comparison at a time, joined with and / or")
        column = self._column(node.left)
        op_type = type(node.ops[0])
        right = node.comparators[0]
        if op_type in (ast.In, ast.NotIn):
            if not isinstance(right, (ast.List, ast.Tuple)):
                raise ValueError('"in" needs a list, e.g. country in ["TR", "US"]')
            options = [_constant(item) for item in right.elts]

            def member(row: dict[str, str]) -> bool:
                return any(_compare_cell(operator.eq, row[column], option) for option in options)

            if op_type is ast.NotIn:
                return lambda row: not member(row)
            return member
        op = _COMPARE_OPS.get(op_type)
        if op is None:
            raise ValueError("unsupported comparison; use ==, !=, <, <=, >, >= or in")
        value = _constant(right)
        return lambda row: _compare_cell(op, row[column], value)

    def _contains(self, node: ast.Call) -> Callable[[dict[str, str]], bool]:
        if len(node.args) != 2 or node.keywords:
            raise ValueError('use: name contains "text"')
        column = self._column(node.args[0])
        needle = _constant(node.args[1])
        if not isinstance(needle, str):
            raise ValueError('contains needs quoted text, e.g. name contains "bike"')
        lowered = needle.lower()
        return lambda row: lowered in row[column].lower()


class _ColumnProfile:
    """Bounded profile of one column: exact counts, capped distinct tracking, reservoir for medians."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.filled = 0
        self.numeric = 0
        self.integers = 0
        self.dates = 0
        self.low: float | None = None
        self.high: float | None = None
        self._mean = 0.0
        self._m2 = 0.0
        self._sample: array = array("d")
        self._counts: dict[str, int] = {}
        self._capped = False

    def add(self, value: str) -> None:
        text = value.strip()
        if not text:
            return
        self.filled += 1
        if value in self._counts or len(self._counts) < MAX_DISTINCT:
            self._counts[value] = self._counts.get(value, 0) + 1
        else:
            self._capped = True
        number = _number(text)
        if number is not None and -1e300 < number < 1e300:
            self._add_number(float(number), bool(_INTEGER_TEXT.fullmatch(text)))
        elif _date_parts(text) is not None:
            self.dates += 1

    def _add_number(self, x: float, integer: bool) -> None:
        self.numeric += 1
        self.integers += int(integer)
        self.low = x if self.low is None else min(self.low, x)
        self.high = x if self.high is None else max(self.high, x)
        delta = x - self._mean
        self._mean += delta / self.numeric
        self._m2 += delta * (x - self._mean)
        if len(self._sample) < MEDIAN_EXACT:
            self._sample.append(x)
        else:
            slot = self._rng.randrange(self.numeric)
            if slot < MEDIAN_EXACT:
                self._sample[slot] = x

    def kind(self) -> str:
        if self.filled == 0:
            return "empty"
        if self.numeric == self.filled:
            return "int" if self.integers == self.numeric else "float"
        if self.dates == self.filled:
            return "date"
        return "text"

    def describe(self, name: str) -> str:
        distinct = f"{MAX_DISTINCT}+" if self._capped else str(len(self._counts))
        lines = [f"- {name} ({self.kind()}): {self.filled} non-empty, {distinct} distinct"]
        if self.kind() in ("int", "float"):
            median = statistics.median(self._sample)
            approx = "≈ " if self.numeric > MEDIAN_EXACT else ""
            stdev = math.sqrt(self._m2 / (self.numeric - 1)) if self.numeric > 1 else None
            lines.append(
                f"  min {_fmt_number(self.low)}, max {_fmt_number(self.high)}, mean {_fmt_number(self._mean)}, "
                f"median {approx}{_fmt_number(median)}, stdev {_fmt_number(stdev)}"
            )
        elif self._counts:
            top = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            lines.append("  top: " + ", ".join(f"{value} ({count})" for value, count in top))
        return "\n".join(lines)


def _fmt_number(value: float | None) -> str:
    return "n/a" if value is None else format_math(value)[0]


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: Any) -> str:
        text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
        return text if len(text) <= CELL_CHARS else text[: CELL_CHARS - 1] + "…"

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _profile(table: _Table, rng: random.Random) -> tuple[str, int]:
    profiles = {column: _ColumnProfile(rng) for column in table.columns}
    sample: list[dict[str, str]] = []
    rows = 0
    for row in table.rows():
        rows += 1
        if len(sample) < 5:
            sample.append(row)
        for column, profile in profiles.items():
            profile.add(row[column])
    parts = [f"Rows: {rows}; columns: {len(table.columns)}", ""]
    parts += [profile.describe(name) for name, profile in profiles.items()]
    parts += ["", "First rows:", _markdown(table.columns, [[row[c] for c in table.columns] for row in sample])]
    return "\n".join(parts), rows


class _Metric:
    """One aggregate (count, sum, mean, min, max or median) for a group, exact with Fraction."""

    def __init__(self, func: str, column: str | None, rng: random.Random) -> None:
        self.func = func
        self.column = column
        self._rng = rng
        self.count = 0
        self.skipped = 0
        self._total = Fraction(0)
        self._low: Fraction | None = None
        self._high: Fraction | None = None
        self._values: list[Fraction] = []

    def add(self, row: dict[str, str]) -> None:
        if self.column is None:
            self.count += 1
            return
        cell = row[self.column]
        if self.func == "count":
            self.count += int(bool(cell.strip()))
            return
        number = _number(cell)
        if number is None:
            self.skipped += int(bool(cell.strip()))
            return
        self.count += 1
        self._total += number
        self._low = number if self._low is None else min(self._low, number)
        self._high = number if self._high is None else max(self._high, number)
        if self.func == "median":
            self._keep_sample(number)

    def _keep_sample(self, number: Fraction) -> None:
        if len(self._values) < MEDIAN_GROUP_SAMPLE:
            self._values.append(number)
            return
        slot = self._rng.randrange(self.count)
        if slot < MEDIAN_GROUP_SAMPLE:
            self._values[slot] = number

    def value(self) -> Fraction | None:
        if self.func == "count" or self.column is None:
            return Fraction(self.count)
        if self.count == 0:
            return None
        if self.func == "sum":
            return self._total
        if self.func == "mean":
            return self._total / self.count
        if self.func == "min":
            return self._low
        if self.func == "max":
            return self._high
        return statistics.median(self._values)

    def text(self) -> str:
        value = self.value()
        if value is None:
            return ""
        approx = "≈ " if self.func == "median" and self.count > MEDIAN_GROUP_SAMPLE else ""
        return approx + format_math(value)[0]


class _Reversed:
    """Inverts ordering so a min-heap keeps the smallest items."""

    __slots__ = ("key",)

    def __init__(self, key: Any) -> None:
        self.key = key

    def __lt__(self, other: "_Reversed") -> bool:
        return other.key < self.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reversed) and self.key == other.key


def _sort_key(value: Any) -> tuple[int, Fraction, str]:
    if value is None:
        return (-1, Fraction(0), "")
    if isinstance(value, Fraction):
        return (0, value, "")
    number = _number(value)
    if number is not None:
        return (0, number, "")
    return (1, Fraction(0), str(value))


def _match_name(name: str, options: list[str]) -> str | None:
    if name in options:
        return name
    matches = [option for option in options if option.lower() == name.lower()]
    return matches[0] if len(matches) == 1 else None


def _parse_sort(text: str, options: list[str]) -> tuple[str, bool]:
    parts = text.split()
    descending = len(parts) > 1 and parts[-1].lower() == "desc"
    name = " ".join(parts[:-1]) if len(parts) > 1 and parts[-1].lower() in ("asc", "desc") else text.strip()
    found = _match_name(name, options)
    if found is None:
        raise ValueError(f"unknown sort column '{name}'; sort by one of: {', '.join(options)}")
    return found, descending


def _metric_label(func: str, column: str | None) -> str:
    return "count" if column is None else f"{func}:{column}"


def _parse_aggregates(text: str, columns: list[str]) -> list[tuple[str, str | None]]:
    specs: list[tuple[str, str | None]] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() == "count":
            specs.append(("count", None))
            continue
        func, separator, column = item.partition(":")
        func = func.strip().lower()
        if func not in AGGREGATES or func == "count" or not separator or not column.strip():
            raise ValueError(
                f"bad aggregate '{item}'; use count, or func:column where func is sum, mean, min, max or median "
                "(e.g. sum:amount)"
            )
        specs.append((func, _resolve_column(column.strip(), columns)))
    if not specs:
        raise ValueError("aggregate is empty; try count or sum:amount")
    return specs


def _aggregate_mode(
    table: _Table,
    where: _Where | None,
    group_by: str | None,
    specs: list[tuple[str, str | None]],
    rng: random.Random,
    limit: int,
    sort_text: str | None,
) -> tuple[str, dict[str, int]]:
    groups: dict[str, list[_Metric]] = {}
    scanned = matched = 0
    for row in table.rows():
        scanned += 1
        if where is not None and not where.matches(row):
            continue
        matched += 1
        key = row[group_by] if group_by else ""
        metrics = groups.get(key)
        if metrics is None:
            if len(groups) >= MAX_GROUPS:
                raise ValueError(
                    f"too many groups (over {MAX_GROUPS}); add a where filter or group by a column with fewer values"
                )
            metrics = groups[key] = [_Metric(func, column, rng) for func, column in specs]
        for metric in metrics:
            metric.add(row)
    metric_labels = [_metric_label(func, column) for func, column in specs]
    labels = ([group_by] if group_by else []) + metric_labels
    texts = [
        ([key or "(empty)"] if group_by else []) + [metric.text() for metric in metrics]
        for key, metrics in groups.items()
    ]
    raw = [
        ([key] if group_by else []) + [metric.value() for metric in metrics]
        for key, metrics in groups.items()
    ]
    if sort_text:
        sort_label, descending = _parse_sort(sort_text, labels)
    else:
        sort_label, descending = metric_labels[0], True
    index = labels.index(sort_label)
    order = sorted(range(len(raw)), key=lambda i: _sort_key(raw[i][index]), reverse=descending)
    shown = [texts[i] for i in order[:limit]]
    direction = "desc" if descending else "asc"
    parts = [f"{matched} rows matched; groups: {len(groups)}", f"sorted by {sort_label} {direction}"]
    skipped = sum(metric.skipped for metrics in groups.values() for metric in metrics)
    if skipped:
        parts.append(f"note: {skipped} non-numeric value(s) were left out of sum/mean/min/max/median")
    parts += ["", _markdown(labels, shown)]
    return "\n".join(parts), {"rows": scanned, "matched": matched}


def _row_mode(
    table: _Table,
    where: _Where | None,
    columns: str | None,
    sort_text: str | None,
    limit: int,
) -> tuple[str, dict[str, int]]:
    selected = (
        [_resolve_column(name.strip(), table.columns) for name in columns.split(",") if name.strip()]
        if columns
        else table.columns
    )
    sort_column = None
    descending = False
    if sort_text:
        sort_column, descending = _parse_sort(sort_text, table.columns)
    kept: list[dict[str, str]] = []
    heap: list[tuple[Any, int, dict[str, str]]] = []
    scanned = matched = 0
    for row in table.rows():
        scanned += 1
        if where is not None and not where.matches(row):
            continue
        matched += 1
        if sort_column is None:
            if len(kept) < limit:
                kept.append(row)
            continue
        key = _sort_key(row[sort_column])
        heapq.heappush(heap, (key if descending else _Reversed(key), matched, row))
        if len(heap) > limit:
            heapq.heappop(heap)
    if sort_column is not None:
        kept = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    how = f"sorted by {sort_column} {'desc' if descending else 'asc'}" if sort_column else "first rows in file order"
    content = f"{matched} rows matched; showing {len(kept)} ({how})\n\n"
    content += _markdown(selected, [[row[name] for name in selected] for row in kept])
    return content, {"rows": scanned, "matched": matched}


ANALYZE_KEYS = ("path", "columns", "where", "group_by", "aggregate", "sort_by", "limit")


def _analyze(args: dict[str, Any], ctx: Any) -> tuple[str, dict[str, int]]:
    path = ctx.workspace.resolve(args["path"], must_exist=True)
    if not path.is_file():
        raise ValueError(f"'{args['path']}' is not a file")
    if path.stat().st_size > MAX_TABLE_BYTES:
        raise ValueError("table is over 500 MB; filter it into a smaller file first")
    table = _Table(path)
    rng = random.Random(0)
    where = _Where(args["where"], table.columns) if args.get("where") else None
    group_by = _resolve_column(args["group_by"], table.columns) if args.get("group_by") else None
    limit = args.get("limit", 20)
    if args.get("aggregate") or group_by:
        specs = _parse_aggregates(args.get("aggregate") or "count", table.columns)
        return _aggregate_mode(table, where, group_by, specs, rng, limit, args.get("sort_by"))
    if args.get("columns") or where is not None or args.get("sort_by"):
        return _row_mode(table, where, args.get("columns"), args.get("sort_by"), limit)
    content, rows = _profile(table, rng)
    return content, {"rows": rows, "matched": rows}


def _reject_unknown(args: dict[str, Any], allowed: tuple[str, ...]) -> None:
    extra = sorted(set(args) - set(allowed))
    if extra:
        raise ValueError(f"unknown option(s): {', '.join(extra)}")


def _limit(value: Any) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be a whole number from 1 to {MAX_LIMIT}")
    return value


def _validate_calculate(args: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(args, ("expression",))
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression is required, e.g. '15% of 240' or '72 F to C'")
    text = expression.strip()
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is too long (limit {MAX_EXPRESSION_CHARS} characters)")
    return {"expression": text}


def _validate_analyze(args: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(args, ANALYZE_KEYS)
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required, e.g. 'data/sales.csv' (workspace-relative)")
    validated: dict[str, Any] = {
        "path": path.strip(),
        "limit": _limit(args["limit"]) if args.get("limit") is not None else 20,
    }
    for key in ("columns", "where", "group_by", "aggregate", "sort_by"):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        if value.strip():
            validated[key] = value.strip()
    return validated


def _execute_calculate(args: dict[str, Any], ctx: Any) -> Any:
    outcome = evaluate(args["expression"])
    content = f"{outcome['input']} = {outcome['result']}"
    if outcome["detail"]:
        content += "\n" + outcome["detail"]
    return tool_result(True, content, meta={"kind": outcome["kind"], "result": outcome["result"]})


def _execute_analyze(args: dict[str, Any], ctx: Any) -> Any:
    try:
        content, counts = _analyze(args, ctx)
    except csv.Error as exc:
        raise ValueError(f"could not parse the table: {exc}") from exc
    return tool_result(True, content, meta=counts)


CALCULATE = ExtensionTool(
    name="calculate",
    description=(
        "Exact math, unit conversion and date arithmetic. Use this instead of calculating in your head. "
        "Examples: '17.5% of 2340', '(3.2e5*12)/7', 'sqrt(2)^10', '72 F to C', '5 miles in km', "
        "'3.5 GiB to MB', 'days between 2026-01-01 and 2026-10-22', '2026-10-22 + 90 days', "
        "'weekday of 2027-03-14'."
    ),
    parameters=schema(
        {"expression": {"type": "string", "description": "What to compute, e.g. '15% of 240' or '10 km to miles'"}},
        ["expression"],
    ),
    risk="read",
    phase="tool",
    label="Calculating",
    summary=lambda a: f"Calculate {str(a.get('expression', ''))[:80]}",
    validate=_validate_calculate,
    execute=_execute_calculate,
    permission="auto",
    category="core",  # exact math is offered on every turn: small models cannot do arithmetic
)

ANALYZE_TABLE = ExtensionTool(
    name="analyze_table",
    description=(
        "Analyze a CSV/TSV/JSON/JSONL table in the workspace without loading it all into memory: profile "
        "columns, filter rows, group and aggregate. Call it with only a path first to see the columns."
    ),
    parameters=schema(
        {
            "path": {"type": "string", "description": "Workspace-relative table file"},
            "columns": {"type": "string", "description": "Comma-separated columns to show"},
            "where": {"type": "string", "description": 'Row filter, e.g. price > 100 and country == "TR"'},
            "group_by": {"type": "string", "description": "Column to group by"},
            "aggregate": {
                "type": "string",
                "description": '"count", "sum:amount", "mean:price", "min:x", "max:x", "median:x"; '
                'comma-separated for several',
            },
            "sort_by": {"type": "string", "description": '"column" or "column desc"'},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "Rows to show (1-200, default 20)"},
        },
        ["path"],
    ),
    risk="read",
    phase="reading",
    label="Analyzing a table",
    summary=lambda a: f"Analyze {a.get('path', '')}",
    validate=_validate_analyze,
    execute=_execute_analyze,
    permission="auto",
)

TOOLS = (CALCULATE, ANALYZE_TABLE)
