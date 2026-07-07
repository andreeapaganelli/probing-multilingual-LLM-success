"""Answer evaluation utilities for math problems.

Provides functions to extract answers from \\boxed{} markup, normalize them,
and compare predictions against ground truth (including MCQ option matching).

These are shared across datasets wherever rollout
correctness needs to be evaluated.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


OPTION_MARKER_RE = re.compile(
    r"(?:\\textbf\s*\{\s*)?\(([A-E])\)(?:\s*\})?", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Answer extraction & normalization
# ---------------------------------------------------------------------------


def _strip_outer_delimiters(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        depth = 0
        balanced = True
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0 and i < len(text) - 1:
                balanced = False
                break
        if not balanced:
            break
        text = text[1:-1].strip()
    return text


def _normalize_frac_braces(text: str) -> str:
    """Ensure \\frac always uses braces: \\frac12 -> \\frac{1}{2}."""
    # Match \frac followed by two single-char args without braces, e.g. \frac12
    text = re.sub(
        r"\\frac\{([^{}]+)\}([0-9A-Za-z])",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac([0-9A-Za-z])\{([^{}]+)\}",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac([0-9A-Za-z])([0-9A-Za-z])",
        r"\\frac{\1}{\2}",
        text,
    )
    return text


def _normalize_pi_products(text: str) -> str:
    """Canonicalize simple products of a rational coefficient and \\pi."""
    # \frac{27}{1024}\pi and \pi\frac{27}{1024}
    # are equivalent to \frac{27\pi}{1024}.
    text = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}\\pi",
        r"\\frac{\1\\pi}{\2}",
        text,
    )
    text = re.sub(
        r"\\pi\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"\\frac{\1\\pi}{\2}",
        text,
    )
    return text


def _strip_unit_annotations(text: str) -> str:
    """Remove unit-only decorations from an otherwise numeric answer."""
    if not re.search(r"\d", text):
        return text

    stripped = text
    stripped = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", stripped)
    stripped = re.sub(
        r"\\(?:text|mathrm)\s*\{\s*[^{}]*?[A-Za-z][^{}]*?\s*\}"
        r"(?:\s*\^\s*\{?[-+]?\d+\}?)?",
        "",
        stripped,
    )
    stripped = re.sub(r"\\(?:degree|degrees)\b", "", stripped)
    stripped = stripped.strip()

    # Plain unit suffixes after a number, e.g. "36 cm^2" or "24 miles".
    unit_suffix = re.fullmatch(
        r"\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*[A-Za-z°][A-Za-z0-9°/^*.\-\s]*\s*",
        stripped,
    )
    if unit_suffix:
        return unit_suffix.group(1)

    return stripped


def normalize_answer(text: str) -> str:
    """Normalize a math answer string for comparison."""
    if text is None:
        return ""

    normalized = str(text).strip()
    normalized = _strip_outer_delimiters(normalized)
    normalized = normalized.replace("\\dfrac", "\\frac")
    normalized = normalized.replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\displaystyle", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\;", "")
    normalized = normalized.replace("\\ ", " ")
    normalized = normalized.replace("\u2212", "-")
    # \sqrt5 -> \sqrt{5} (brace-less sqrt argument)
    normalized = re.sub(r"\\sqrt([0-9A-Za-z])", r"\\sqrt{\1}", normalized)

    # Strip \textbf{(A)} / \textbf{(A) } MCQ letter markers that models sometimes box alongside the value
    normalized = re.sub(r"\\textbf\s*\{[^{}]*\([A-E]\)[^{}]*\}\s*\\?\s*", "", normalized).strip()

    # Normalize subscript/superscript braces for single chars: _{9}->_9, ^{2}->^2
    normalized = re.sub(r"_\{([0-9])\}", r"_\1", normalized)
    normalized = re.sub(r"\^\{([0-9A-Za-z])\}", r"^\1", normalized)

    # Treat \text{and} / \text{ and } as a comma separator for multi-value answers
    normalized = re.sub(r"\\text\s*\{\s*and\s*\}", ",", normalized)

    normalized = _strip_unit_annotations(normalized)

    # Strip percent signs and dollar signs
    normalized = normalized.replace("\\%", "")
    normalized = normalized.replace("%", "")
    normalized = normalized.replace("\\$", "")

    text_match = re.fullmatch(r"\\text\s*\{\s*([^{}]+?)\s*\}", normalized)
    if text_match:
        normalized = text_match.group(1)

    normalized = normalized.strip().strip(".").strip()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"^[\(\[]|[\)\]]$", "", normalized)

    # Remove thousand-separator commas from numbers (e.g. "11,400" -> "11400").
    # Only strip commas followed by exactly 3 digits to avoid clobbering
    # coordinate-pair commas like "1,4.5" → which would wrongly become "14.5".
    normalized = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", normalized)

    # Normalize \frac brace formatting
    normalized = _normalize_frac_braces(normalized)
    normalized = _normalize_pi_products(normalized)

    # Strip leading zeros from integers (042 -> 42) but not "0" or "0.5"
    normalized = re.sub(r"^(-?)0+(\d+)$", r"\1\2", normalized)

    # Strip trailing .0 from decimals (28.0 -> 28)
    normalized = re.sub(r"^(-?\d+)\.0+$", r"\1", normalized)

    if re.fullmatch(r"[A-Ea-e]", normalized):
        return normalized.upper()

    return normalized


def extract_last_boxed_content(text: str) -> str | None:
    """Extract the content of the last non-empty \\boxed{...} in *text*.

    Iterates backwards so that empty \\boxed{} placeholders quoted from the
    prompt template (e.g. "write your answer in \\boxed{}") do not shadow a
    real answer that appears earlier in the rollout.
    """
    if not text:
        return None

    for m in reversed(list(re.finditer(r"\\boxed", text))):
        start = m.start() + len("\\boxed")
        while start < len(text) and text[start].isspace():
            start += 1

        if start >= len(text):
            continue

        if text[start] == "{":
            depth = 0
            end = start
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth == 0:
                    end = i
                    break
            content = text[start + 1 : end].strip()
            if content:
                return content
            # empty \boxed{} — keep searching backwards
            continue

        token_match = re.match(r"([A-Za-z]|[-+]?\d+(?:/\d+)?(?:\.\d+)?)", text[start:])
        if token_match:
            return token_match.group(1)

    return None


# ---------------------------------------------------------------------------
# MCQ option handling
# ---------------------------------------------------------------------------


def _clean_option_text(option_text: str) -> str:
    cleaned = option_text.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"^\s*[\\\}\{\$,:;]+", "", cleaned)
    cleaned = re.sub(r"\s*(?:\\qquad|\\quad)\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.split(r"\bEnter the letter\b", cleaned, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\bChoose the correct\b", cleaned, flags=re.IGNORECASE)[0]
    return cleaned.strip(" $")


def parse_mcq_options(problem_text: str) -> dict[str, str]:
    """Parse MCQ option markers ``(A)``..``(E)`` from a problem statement."""
    if not problem_text:
        return {}

    matches = list(OPTION_MARKER_RE.finditer(problem_text))
    if len(matches) < 2:
        return {}

    options: dict[str, str] = {}
    for i, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(problem_text)
        option_text = problem_text[start:end]
        option_text = _clean_option_text(option_text)
        if option_text:
            options[letter] = option_text

    if len(options) < 2:
        return {}

    return options


def infer_correct_letter(
    ground_truth: str, options: dict[str, str]
) -> str | None:
    """Given ground truth and parsed MCQ options, return the correct letter."""
    if not options:
        return None

    gt_norm = normalize_answer(ground_truth)
    if re.fullmatch(r"[A-E]", gt_norm):
        return gt_norm

    for letter, option_value in options.items():
        opt_norm = normalize_answer(option_value)
        if opt_norm == gt_norm or _numerically_equal(opt_norm, gt_norm):
            return letter

    return None


def expand_prediction_forms(
    pred_norm: str, options: dict[str, str]
) -> set[str]:
    """Expand a normalized prediction into all equivalent forms (letter ↔ value)."""
    forms = {pred_norm}
    if not pred_norm or not options:
        return forms

    if re.fullmatch(r"[A-E]", pred_norm):
        option_value = options.get(pred_norm)
        if option_value is not None:
            forms.add(normalize_answer(option_value))
        return forms

    for letter, option_value in options.items():
        if normalize_answer(option_value) == pred_norm:
            forms.add(letter)

    return forms


_INEQ_VAR_OP_BOUND_RE = re.compile(
    r"^[a-zA-Z](<=?|>=?|\\leq?|\\geq?)(.+)$"
)
_INEQ_BOUND_OP_VAR_RE = re.compile(
    r"^(.+?)(<=?|>=?|\\leq?|\\geq?)[a-zA-Z]$"
)


def _inequality_to_interval_norm(norm: str) -> str | None:
    r"""Convert a normalized single-variable inequality to its interval string.

    Examples (after normalize_answer has already run):
      'k<-\\frac{1}{5}' → '-\\infty,-\\frac{1}{5}'
      'x>=3'            → '3,\\infty'
      '3>x'             → '-\\infty,3'
    Returns None if *norm* does not look like a simple inequality.
    """
    m = _INEQ_VAR_OP_BOUND_RE.match(norm)
    if m:
        op, bound = m.group(1), m.group(2)
        if op in ("<", "<=", r"\le", r"\leq"):
            return rf"-\infty,{bound}"
        if op in (">", ">=", r"\ge", r"\geq"):
            return rf"{bound},\infty"
    m = _INEQ_BOUND_OP_VAR_RE.match(norm)
    if m:
        bound, op = m.group(1), m.group(2)
        # "bound < var" means var > bound → (bound, +∞)
        if op in ("<", "<=", r"\le", r"\leq"):
            return rf"{bound},\infty"
        # "bound > var" means var < bound → (−∞, bound)
        if op in (">", ">=", r"\ge", r"\geq"):
            return rf"-\infty,{bound}"
    return None


def _split_top_level_equals(text: str) -> list[str]:
    """Split on equals signs that are not nested inside braces."""
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas not nested inside braces."""
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def normalized_answer_forms(text: str) -> set[str]:
    r"""Return conservative normalized variants of a predicted answer.

    Besides the raw normalized answer, this handles two common model-output
    shapes: boxed values with units and boxed equality chains such as
    ``120 \times 4 = 480``.  It intentionally does not extract arbitrary
    numbers from expressions, so ``3680 \cdot 7^{20}`` will not match ``3680``.
    """
    forms = {normalize_answer(text)}
    parts = _split_top_level_equals(str(text))
    if len(parts) > 1:
        rhs = parts[-1].strip()
        if rhs:
            forms.add(normalize_answer(rhs))
    return {form for form in forms if form}


# ---------------------------------------------------------------------------
# Rollout scoring
# ---------------------------------------------------------------------------


def _to_numeric(s: str) -> float | None:
    """Try to convert a normalized answer string to a float.

    Handles plain numbers, \\frac{a}{b}, and a/b notation.
    Returns None if conversion fails.
    """
    if not s:
        return None
    # Plain number
    try:
        return float(s)
    except ValueError:
        pass
    # \frac{a}{b}
    m = re.match(r"^-?\\frac\{([^{}]+)\}\{([^{}]+)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # a/b
    m = re.match(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # a:b ratio notation (e.g. "3:5" == 3/5)
    m = re.match(r"^(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _numerically_equal(a: str, b: str, rel_tol: float = 1e-9) -> bool:
    """Check if two normalized answer strings are numerically equal."""
    va = _to_numeric(a)
    vb = _to_numeric(b)
    if va is not None and vb is not None:
        if va == vb == 0:
            return True
        return abs(va - vb) <= rel_tol * max(abs(va), abs(vb))
    # Tuple / coordinate-pair comparison: "1,4.5" == "1,\frac{9}{2}"
    parts_a = _split_top_level_commas(a)
    parts_b = _split_top_level_commas(b)
    if len(parts_a) > 1 and len(parts_a) == len(parts_b):
        return all(
            _numerically_equal(pa.strip(), pb.strip())
            for pa, pb in zip(parts_a, parts_b)
        )
    return False


def evaluate_rollout_text(
    text: str,
    ground_truth: str,
    options: dict[str, str] | None = None,
    correct_letter: str | None = None,
) -> int:
    """Score a single rollout: 1 if correct, 0 otherwise.

    Extracts the last ``\\boxed{...}`` from *text* and compares against
    *ground_truth*, optionally considering MCQ option equivalences.
    Uses string comparison first, then falls back to numeric comparison.
    """
    if options is None:
        options = {}

    boxed = extract_last_boxed_content(text)
    if boxed is None:
        return 0

    pred_forms = normalized_answer_forms(boxed)
    gt_forms = normalized_answer_forms(ground_truth)
    gt_norm = normalize_answer(ground_truth)
    expanded_pred_forms: set[str] = set()
    for pred_form in pred_forms:
        expanded_pred_forms.update(expand_prediction_forms(pred_form, options))

    if correct_letter is not None:
        if correct_letter in expanded_pred_forms or gt_forms & expanded_pred_forms:
            return 1
        correct_option = options.get(correct_letter)
        if correct_option is not None and normalize_answer(correct_option) in expanded_pred_forms:
            return 1
        # Numeric fallback for MCQ
        for pred_form in pred_forms:
            if any(_numerically_equal(pred_form, gf) for gf in gt_forms):
                return 1
        return 0

    if gt_forms & expanded_pred_forms:
        return 1

    # Numeric fallback: decimal vs fraction, unsimplified fractions, etc.
    if any(_numerically_equal(pred_form, gf) for pred_form in pred_forms for gf in gt_forms):
        return 1

    # Inequality ↔ interval equivalence: "k<-\frac{1}{5}" == "(-\infty,-\frac{1}{5})"
    for pred_form in pred_forms:
        interval = _inequality_to_interval_norm(pred_form)
        if interval is not None and interval in gt_forms:
            return 1
    for gf in gt_forms:
        gt_interval = _inequality_to_interval_norm(gf)
        if gt_interval is not None and gt_interval in expanded_pred_forms:
            return 1

    return 0


def compute_success_rate(scores: Iterable[int]) -> float:
    """Fraction of correct rollouts (mean of binary scores)."""
    scores = list(scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f_value = float(value)
        import math
        if math.isnan(f_value):
            return default
        return f_value
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
