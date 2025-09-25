# This file is mainly from Math-Verify, modified by yhhu

import re
from functools import lru_cache
from itertools import groupby
import sympy
from sympy import Basic, MatrixBase, Number
from sympy.parsing import parse_expr
from typing import Literal, Union
from .utils import timeout
from .latex2sympy2_extended.sets import FiniteSet
from .latex2sympy2_extended.latex2sympy2 import (
    NormalizationConfig,
    normalize_latex,
    latex2sympy,
)

# Currently, we only use the last \\boxed{} to extract only one answer
# If there is need to extract multiple answer, modify the regex in make_latex_env_pattern
# If there is need to extract answers according to multiple format (such as 'the answer is: ...'), modify 'lazy_latex_regex'

complex_number_pattern = re.compile(
    r"""
    # Complex number indicators
    \\mathbb\{C\}|        # Complex number set ℂ
    \\i\b|                # Complex i
    \bi\b|                # Standalone i
    \\text\{i\}|          # Text i
    \\mathrm\{i\}|        # Roman i
    \\imath\b|            # Alternative i notation

    # Matrix operations
    \\det|                # Determinant
    \\operatorname\{tr\}| # Trace
    \\operatorname\{rank\}| # Rank
    \\text\{rank\}|
    \\arg\{|              # Complex argument
    \\Re\{|               # Real part
    \\Im\{|               # Imaginary part
    \\operatorname\{Re\}| # Real part alternate
    \\operatorname\{Im\}| # Imaginary part alternate
    \\text\{Re\}|         # Real part text
    \\text\{Im\}          # Imaginary part text
""",
    re.VERBOSE,
)


def should_treat_as_complex(latex_str: str) -> bool:
    """
    Returns True if the latex string likely contains complex numbers, matrices, or vectors.
    """

    return bool(complex_number_pattern.search(latex_str))


#########################################
# Initial extraction from model response
#########################################

def extract_box_str(resp: str) -> str:
    resp = resp.replace("\u043a\u0438", "")
    pred_str = ""
    if "boxed" in resp:
        ans = resp.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred_str = a

    return pred_str



##########################################
# Expression and Latex and Number Parsing
##########################################

def make_latex_env_pattern(prefix: str = "") -> str:
    """Creates a LaTeX environment pattern with uniquely prefixed group names.

    Args:
        prefix (str): Prefix to add to group names to make them unique

    Returns:
        str: Regex pattern for matching LaTeX environments with percent suffix
    """
    percent_re_group = rf"(?P<{prefix}percent>(?:\\?%|[Pp]ercent|[Pp]ercentage|[Pp]ct))"

    # Define base content patterns
    display_dollar_content = r"(?:[^$]|\$(?!\$))"
    # Either \ not followed by ] or everything but \
    display_content_bracket = r"(?:[^\\]|\\(?!\]))"
    inline_dollar_content = r"(?:\\[$]|[^\n$])"
    inline_content_parenthesis = r"(?:[^\\\n]|\\(?!\)))"
    inline_content_bracket = r"[^\n\]\[]"

    display_dollar_content = rf"{display_dollar_content}+?"
    display_content_bracket = rf"{display_content_bracket}+?"
    inline_dollar_content = rf"{inline_dollar_content}+?"
    inline_content_parenthesis = rf"{inline_content_parenthesis}+?"
    inline_content_bracket = rf"{inline_content_bracket}+?"

    # display_dollar_content = rf"{display_dollar_content}*?\\boxed{{{display_dollar_content}+?}}{display_dollar_content}*?"
    # display_content_bracket = rf"{display_content_bracket}*?\\boxed{{{display_content_bracket}+?}}{display_content_bracket}*?"
    # inline_dollar_content = rf"{inline_dollar_content}*?\\boxed{{{inline_dollar_content}+?}}{inline_dollar_content}*?"
    # inline_content_parenthesis = rf"{inline_content_parenthesis}*?\\boxed{{{inline_content_parenthesis}+?}}{inline_content_parenthesis}*?"
    # inline_content_bracket = rf"{inline_content_bracket}*?\\boxed{{{inline_content_bracket}+?}}{inline_content_bracket}*?"

    # Build list of regex patterns
    patterns = [
        # Display math environments (allow multiline)
        rf"(?<!\\)\$\$(?P<{prefix}latexDisplayDollar>{display_dollar_content})(?<!\\)\$\$",
        rf"(?<!\\)\\\[(?P<{prefix}latexDisplayBracket>{display_content_bracket})(?<!\\)\\\]",
        # Inline math environments (single line only)
        rf"(?<!\\|\d)\$(?P<{prefix}latexInlineDollar>{inline_dollar_content})(?<!\\)\$",
        rf"(?<!\\)\\\((?P<{prefix}latexInlineParenthesis>{inline_content_parenthesis})(?<!\\)\\\)",
        rf"\s\[(?P<{prefix}latexInlineBracket>{inline_content_bracket})\]\s",
    ]

    simple_number = r"-?\d+(?:[.,]\d+)?"
    patterns.append(rf"(?P<{prefix}latexFraction>-?\\frac{{{simple_number}}}{{{simple_number}}})")

    # Join patterns with | and wrap in parentheses
    latex_env_re = rf"(?:(?:{'|'.join(patterns)})\s*{percent_re_group}?)"

    return latex_env_re


@lru_cache(maxsize=1)
def lazy_latex_regex(
        latex_config: NormalizationConfig,
) -> list[tuple[re.Pattern[str], int]]:
    # Pattern for multiple latex environments connected by and/or
    # Create patterns for up to 5 connected expressions
    first_latex_group = make_latex_env_pattern('first_')
    next_groups = ''.join([rf"(?:\s*(?:,?\s?and|,?\s?or|,)\s*{make_latex_env_pattern(f'next{i}_')})?" for i in range(1, 6)])

    latex_envs_re = rf"(?:{first_latex_group}{next_groups})"
    regexes: list[tuple[str, int]] = [(latex_envs_re, 0)]
    # for latex_re in [latex_envs_re]:
    #     final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{latex_re}\.?\s?I hope"

    return [(re.compile(pattern, re.DOTALL), priority) for pattern, priority in regexes]


@lru_cache(maxsize=1)
def lazy_expr_regex(
        expr_config: NormalizationConfig,
) -> list[tuple[re.Pattern[str], int]]:
    number_re = (
        # Format 1: Numbers with thousands separators (e.g., "1,234.56" or "1 234.56")
        r"(?<!\d)(?:"
        r"(?P<integer1>-?[1-9]\d{0,2}(?:[ ,]\d{3})+)(?P<decimal1>\.\d+)?|"
        # Format 2: Simple numbers with decimal point or comma (e.g., "123.45" or "123,45")
        r"(?P<integer2>-?\d+)(?P<decimal2>[.,]\d+)|"
        # Format 3: Decimal part only (e.g., ".123")
        r"(?P<decimal3>\.\d+)|"
        # Format 4: Integer only (e.g., "123")
        r"(?P<integer3>-?\d+)"
        r")(?P<percent>\s*(?:%|[Pp]ercent|\s*[Pp]ercentage|\s*[Pp]ct))?"
    )

    # Expressions such as 1/2
    operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
    operators_re = "".join(operators)
    all_expr_chars = r"[\d\.\s" + operators_re + r"]"
    # Expression should have at minimum at least one operator and must start with a digit
    expr_re = (
        rf"(?P<expr>-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?)"
    )

    # Punctuation regexes
    full_stop_re = rf"\."
    comma_re = rf","
    colon_re = rf":"
    space_re = rf"\s"

    currency_units = re.escape("$€£¥₹₽₪₩₫฿₡₢₣₤₥₦₧₨₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿")
    expr_prefix_re = rf"(?:^|{space_re}|\=)(?:\*\*)?"
    expr_suffix_re = (
        rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|\)|\$|$)"
    )
    # Expressions must be prefixed and suffixed while, digits don't need suffix and can have currency units preceeded, this is to ensure
    # That we can extract stuff like $100 or 100m2, while we don't extract XDY2K as 2
    expr_with_anchors = rf"(?:{expr_prefix_re}{expr_re}{expr_suffix_re})"
    number_with_anchors = rf"(?:{expr_prefix_re}[{currency_units}]?{number_re})"
    expr_or_number = rf"(?:{expr_with_anchors}|{number_with_anchors})"

    regexes: list[tuple[str, int]] = [(expr_or_number, 50)]
    # for latex_re in [latex_envs_re]:
    #     final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{latex_re}\.?\s?I hope"

    return [(re.compile(pattern, re.DOTALL), priority) for pattern, priority in regexes]


# Small cache, to cache repeated calls invalid parsing
@lru_cache(maxsize=20)
def parse_latex_cached(latex: str):
    # First try to parse the latex as is
    try:
        return latex2sympy(
            latex, is_real=not should_treat_as_complex(latex), convert_degrees=False, normalization_config=None
        )
    except Exception as e:
        # If that fails, try to parse just the last equation
        last_eq_latex = get_last_eq(latex)
        if last_eq_latex != latex:
            return latex2sympy(
                last_eq_latex, is_real=not should_treat_as_complex(last_eq_latex), convert_degrees=False, normalization_config=None
            )
        else:
            raise e


@lru_cache(maxsize=20)
def parse_expr_cached(expr: str):
    return parse_expr(expr, evaluate=False)


def convert_to_pct(number: Number):
    return sympy.Mul(number, sympy.Rational(1, 100), evaluate=False)


equation_split_regex = re.compile(r"(?<!\\|\<|\!|\>)=")
def get_last_eq(latex: str):
    # This is to ensure that a=1,b=2 won't be split
    if not "," in latex and not ";" in latex:
        eq_parts = equation_split_regex.split(latex)
        # We only shorten if there are more than 2 parts, otherwise we keep equation as is
        if len(eq_parts) > 2:
            return eq_parts[-1]
    return latex


def extract_latex(match: re.Match) -> tuple[Union[sympy.Expr, str, None], str]:
    latex_exprs = []
    latex_strs = []

    # Get all latex groups (both first_ and nextN_ prefixes)
    first_latex_group = next(
        ((val, name) for name, val in match.groupdict().items() if name.startswith("first_latex") and val),
        None
    )
    # Get all nextN_ groups
    next_latex_groups = [
        next(
            ((val, name) for name, val in match.groupdict().items() if name.startswith(f"next{i}_latex") and val),
            None
        )
        for i in range(1, 6)
    ]

    all_latex = list(filter(lambda x: x is not None, [first_latex_group] + next_latex_groups))

    for latex, name in all_latex:
        name_without_prefix = name.split('_')[0]
        is_percentage = True if match.groupdict().get(f"{name_without_prefix}_percent") else False

        # Use modified config if group name is 'boxed'
        # group_name = name.split('_')[1] if len(name.split('_')) > 1 else None
        # config = latex_config.normalization_config
        # if group_name == 'latexBoxed':
        #     config = replace(config, boxed="last")  # Use replace to modify single field

        normalized_latex = normalize_latex(
            latex,
            config=NormalizationConfig(), # TODO: add specific config if needed
        )
        latex_strs.append(normalized_latex)
        try:
            parsed_latex = parse_latex_cached(normalized_latex)
            if is_percentage:
                parsed_latex = convert_to_pct(parsed_latex)
            latex_exprs.append(parsed_latex)
        except:
            latex_exprs.append(None)
            pass

    if not latex_exprs:
        return None, ""

    # If we have multiple expressions and all of them are parsed, wrap them in a Tuple
    if len(latex_exprs) > 1 and all(expr is not None for expr in latex_exprs):
        # To handle solution is: 1,2 and 3
        all_elements = []
        for expr in latex_exprs:
            if isinstance(expr, FiniteSet):
                all_elements.extend(expr.args)
            else:
                all_elements.append(expr)
        return FiniteSet(*all_elements), " and ".join(latex_strs)

    # Otherwise return the single expression
    return latex_exprs[0], latex_strs[0]



def extract_expr(match: re.Match) -> tuple[Union[str, sympy.Expr, Number, None], str]:
    # First combine the number
    groups = match.groupdict()
    # Expr group will always exist because every regex has it
    expr = groups.get("expr", "")
    integer = next(
        (val for name, val in groups.items() if name.startswith("integer") and val), ""
    )
    decimal = next(
        (val for name, val in groups.items() if name.startswith("decimal") and val), ""
    )

    is_percentage = True if groups.get("percent", None) else False

    if integer or decimal:
        # This makes sure we can convert numbers like 0001 to 1. Do note that this can convert 0 to '', so we assume an empty string was 0 and convert it back afterwards.
        integer = integer.translate(str.maketrans("", "", ", ")).lstrip("0")
        if len(integer) == 0:
            integer = "0"

        decimal = decimal.replace(",", ".")
        number_str = f"{integer}{decimal}"
        number = Number(number_str)

        if is_percentage:
            number = convert_to_pct(number)
        return number, number_str

    # Otherwise just return the expression
    # Remove new lines and spaces
    if expr:
        try:
            return (
                parse_expr_cached(expr.replace("\n", " ").replace("^", "**")),
                expr,
            )
        except:
            pass
    return None, expr


def extract_match(match: re.Match, target_type: str) -> tuple[Union[Basic,MatrixBase,str,None], str]:
    if target_type == 'latex':
        return extract_latex(match)
    elif target_type == 'expr':
        return extract_expr(match)


def extract(pred_str: str, skip_unit: bool = True):
    parse_config = NormalizationConfig(units=not skip_unit)
    latex_regexes = lazy_latex_regex(parse_config)
    expr_regexes = lazy_expr_regex(parse_config)
    all_patterns = []
    all_patterns.extend((pattern, "latex", priority) for pattern, priority in latex_regexes)
    # TODO: figure out if there is need to extract expression
    # all_patterns.extend((pattern, "expr", priority) for pattern, priority in expr_regexes)

    extracted_predictions = []
    fallbacks = []

    # find by re matching
    match_found = False
    sorted_patterns = sorted(all_patterns, key=lambda x: x[2])
    grouped_patterns = list((gr, list(val)) for gr, val in groupby(sorted_patterns, key=lambda x: x[2]))
    for _, patterns_group in grouped_patterns:
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred_str)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(
            matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
        )

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            extracted_match, str_fallback = extract_match(match, target_type)

            match_found = True
            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break


        # If we extracted something or found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions:
            break

    normalized_orig_pred = normalize_latex(
        pred_str,
        config=parse_config,
    )
    # Number check
    try:
        number_pred = Number(normalized_orig_pred)
        extracted_predictions.append(number_pred)
    except:
        try:
            parsed_orig_pred = parse_latex_cached(normalized_orig_pred)
            if re.match(r"\\?%|[Pp]ercent|[Pp]ercentage|[Pp]ct", normalized_orig_pred) is not None:
                parsed_orig_pred = convert_to_pct(parsed_orig_pred)
            extracted_predictions.append(parsed_orig_pred)
        except:
            pass
    fallbacks.append(normalized_orig_pred)

    extracted_predictions.extend(fallbacks)

    return extracted_predictions


def parse(pred_str: str, skip_unit: bool = True, time_limit: int = 5):
    if "\\boxed" in pred_str:
        pred_str = extract_box_str(pred_str)
    try:
        return timeout(timeout_seconds=time_limit)(extract)(pred_str, skip_unit)
    except:
        return []

if __name__ == '__main__':
    print(parse("2√3"))