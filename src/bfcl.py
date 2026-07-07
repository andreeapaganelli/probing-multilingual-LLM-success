from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BFCL_REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_DEFAULT_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")
BFCL_CATEGORY_FILE_STEMS = {
    "simple_python": "simple",
    "simple": "simple",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "live_simple": "live_simple",
    "live_multiple": "live_multiple",
    "live_parallel": "live_parallel",
    "live_parallel_multiple": "live_parallel_multiple",
    "live_irrelevance": "live_irrelevance",
    "live_relevance": "live_relevance",
}


@dataclass(frozen=True)
class BFCLExample:
    example_id: str
    user_prompt: str
    functions: list[dict[str, Any]]
    ground_truth: Any
    category: str
    split: str


def stable_unit_interval(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def assign_split(example_id: str, seed: int = 42) -> str:
    u = stable_unit_interval(example_id, seed=seed)
    if u < 0.70:
        return "train"
    if u < 0.85:
        return "validation"
    return "test"


def safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_user_prompt(question: Any) -> str:
    question = safe_json_loads(question)
    if isinstance(question, list):
        cursor = question
        while isinstance(cursor, list) and cursor:
            cursor = cursor[0]
        if isinstance(cursor, dict):
            return str(cursor.get("content", ""))
    if isinstance(question, dict):
        return str(question.get("content", ""))
    return str(question)


def build_bfcl_system_prompt(example: dict[str, Any]) -> str:
    functions_json = json.dumps(example["functions"], ensure_ascii=False, indent=2)
    return (
        "You are an expert in composing functions. You are given a question and "
        "a set of possible functions. Based on the question, you will need to "
        "make one or more function/tool calls to achieve the purpose. If none "
        "of the functions can be used, point it out. If the given question lacks "
        "the parameters required by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the "
        "format of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
        "func_name2(params)]. You SHOULD NOT include any other text in the response.\n\n"
        "Here is a list of functions in JSON format that you can invoke.\n"
        f"{functions_json}"
    )


def format_bfcl_qwen_manual_prompt(example: dict[str, Any]) -> str:
    """Reproduce BFCL's default system prompt and manual QwenHandler wrapper."""
    functions_json = json.dumps(example.get("functions", []), indent=4)
    system = (
        "You are an expert in composing functions.You are given a question and a set "
        "of possible functions. Based on the question, you will need to make one or "
        "more function/tool calls to achieve the purpose. If none of the functions "
        "can be used, point it out. If the given question lacks the parameters "
        "required by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the format "
        "of [func_name1(params_name1=params_value1, params_name2=params_value2...), "
        "func_name2(params)].  You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by the "
        "user within the current turn. Continue to output functions to call until you "
        "have fulfilled the user's request to the best of your ability. Once you have "
        "no more functions to call, the system will consider the current turn complete "
        "and proceed to the next turn or task.\n\n"
        "Here is a list of functions in json format that you can invoke.\n"
        f"{functions_json}\n"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{example['user_prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_bfcl_chat_prompt(
    example: dict[str, Any],
    tokenizer=None,
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    use_fc_template: bool = False,
) -> str:
    """Build an inference prompt for a BFCL example.

    ``enable_thinking`` controls the Qwen thinking-block prefix:

    * ``None``  (default) – do not pass the kwarg; the tokenizer uses its own
      default (model decides whether to think).  Matches the BFCL-repo default
      ``QwenHandler`` behaviour where ``<think>`` blocks may appear but are
      stripped from model responses by the handler.
    * ``False`` – pass ``enable_thinking=False``, which prepends
      ``<think>\\n\\n</think>\\n\\n`` and forces the model to skip thinking.
    * ``True``  – pass ``enable_thinking=True``, which prepends ``<think>\\n``
      and forces the model to think.

    When ``use_fc_template=True`` the function specs are passed as OpenAI-style
    ``tools=`` via the tokenizer's chat template (Harmony FC mode for GPT-OSS).
    """
    if use_fc_template:
        # FC mode: pass tools via tokenizer, skip BFCL's text system prompt.
        messages = [{"role": "user", "content": str(example["user_prompt"])}]
        tools = [
            {"type": "function", "function": f}
            for f in example.get("functions", [])
        ]
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": reasoning_effort,
        }
        if tools:
            kwargs["tools"] = tools
        if tokenizer is None:
            return f"User: {messages[0]['content']}\n\nAssistant:"
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    system = build_bfcl_system_prompt(example)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": str(example["user_prompt"])},
    ]
    if tokenizer is None:
        return f"System: {system}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        # Explicitly pass only when caller wants to override the default.
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    # enable_thinking=None → BFCL default: let the tokenizer decide.
    return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_bfcl_data(
    output_path: Path,
    *,
    categories: list[str] | tuple[str, ...] = BFCL_DEFAULT_CATEGORIES,
    cache_dir: str = "data/hf_cache",
    local_data_root: str | Path | None = None,
    seed: int = 42,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    examples: list[dict[str, Any]] = []
    categories = [c.strip() for c in categories if c.strip()]
    for category in categories:
        file_stem = BFCL_CATEGORY_FILE_STEMS.get(category)
        if file_stem is None:
            raise ValueError(
                f"Unsupported BFCL category {category!r}. "
                f"Supported first-version categories: {sorted(BFCL_CATEGORY_FILE_STEMS)}"
            )
        is_v4 = category.startswith("live_")
        version = "v4" if is_v4 else "v3"
        data_file = f"BFCL_{version}_{file_stem}.json"
        answer_file = f"possible_answer/BFCL_{version}_{file_stem}.json"
        if local_data_root is not None:
            root = Path(local_data_root)
            data_path = root / data_file
            answer_path = root / answer_file
        else:
            data_path = Path(hf_hub_download(BFCL_REPO_ID, data_file, repo_type="dataset", cache_dir=cache_dir))
            answer_path = Path(hf_hub_download(BFCL_REPO_ID, answer_file, repo_type="dataset", cache_dir=cache_dir))

        answers_by_id = {}
        if answer_path.exists():
            with open(answer_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    answers_by_id[str(row["id"])] = row.get("ground_truth")

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                example_id = str(row["id"])
                record = {
                    "example_id": example_id,
                    "user_prompt": extract_user_prompt(row.get("question")),
                    "functions": row.get("function", []),
                    "ground_truth": answers_by_id.get(example_id),
                    "category": "simple_python" if category == "simple" else category,
                    "split": assign_split(example_id, seed=seed),
                }
                examples.append(record)

    examples = sorted(examples, key=lambda x: (x["category"], x["example_id"]))
    if max_examples is not None:
        examples = examples[:max_examples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return examples


def load_processed_bfcl_examples(
    path: Path | str,
    *,
    split: str | None = None,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                continue
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def _normalise_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().strip("\"'")
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _normalise_arg_values(args: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for key, value in args.items():
        if isinstance(value, list):
            out[str(key)] = [_normalise_scalar(v) for v in value]
        else:
            out[str(key)] = [_normalise_scalar(value)]
    return out


def _call_from_ast(node: ast.AST) -> dict[str, dict[str, list[Any]]] | None:
    if not isinstance(node, ast.Call):
        return None
    try:
        name = ast.unparse(node.func)
    except Exception:
        return None
    args: dict[str, Any] = {}
    for i, arg in enumerate(node.args):
        try:
            args[f"arg_{i}"] = ast.literal_eval(arg)
        except Exception:
            try:
                args[f"arg_{i}"] = ast.unparse(arg)
            except Exception:
                args[f"arg_{i}"] = ""
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            try:
                args[kw.arg] = ast.unparse(kw.value)
            except Exception:
                args[kw.arg] = ""
    return {name: _normalise_arg_values(args)}


def parse_model_calls(text: str) -> list[dict[str, dict[str, list[Any]]]]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"^```(?:python|json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    candidates = [text]
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, list):
            candidates = [str(x) if not isinstance(x, dict) else json.dumps(x) for x in parsed_json]
        elif isinstance(parsed_json, dict):
            if "name" in parsed_json and "arguments" in parsed_json:
                return [{str(parsed_json["name"]): _normalise_arg_values(parsed_json.get("arguments") or {})}]
            candidates = [json.dumps(parsed_json)]
    except Exception:
        pass

    calls = []
    for candidate in candidates:
        try:
            tree = ast.parse(candidate, mode="eval")
            body = tree.body
            if isinstance(body, (ast.List, ast.Tuple)):
                for elem in body.elts:
                    call = _call_from_ast(elem)
                    if call is not None:
                        calls.append(call)
            else:
                call = _call_from_ast(body)
                if call is not None:
                    calls.append(call)
        except Exception:
            for match in re.finditer(r"([A-Za-z_][\w.]*\s*\([^()]*\))", candidate):
                try:
                    tree = ast.parse(match.group(1), mode="eval")
                    call = _call_from_ast(tree.body)
                    if call is not None:
                        calls.append(call)
                except Exception:
                    continue
    return calls


def normalise_ground_truth(ground_truth: Any) -> list[dict[str, dict[str, list[Any]]]]:
    ground_truth = safe_json_loads(ground_truth)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    out = []
    for item in ground_truth:
        if not isinstance(item, dict):
            continue
        normalised = {}
        for fn_name, args in item.items():
            normalised[str(fn_name)] = _normalise_arg_values(args if isinstance(args, dict) else {})
        out.append(normalised)
    return out


def bfcl_ast_correct(generation: str, ground_truth: Any) -> bool:
    predicted = parse_model_calls(generation)
    expected = normalise_ground_truth(ground_truth)
    if len(predicted) != len(expected):
        return False
    unmatched = list(expected)
    for pred in predicted:
        match_idx = next(
            (i for i, exp in enumerate(unmatched) if _call_matches(pred, exp)),
            None,
        )
        if match_idx is None:
            return False
        unmatched.pop(match_idx)
    return not unmatched


def _call_matches(
    predicted: dict[str, dict[str, list[Any]]],
    expected: dict[str, dict[str, list[Any]]],
) -> bool:
    if set(predicted.keys()) != set(expected.keys()):
        return False
    for fn_name, expected_args in expected.items():
        predicted_args = predicted.get(fn_name, {})
        for arg_name, allowed_values in expected_args.items():
            if arg_name not in predicted_args:
                if "" in allowed_values or None in allowed_values:
                    continue
                return False
            pred_values = predicted_args[arg_name]
            if not all(value in allowed_values for value in pred_values):
                return False
        required_predicted = {
            k: v for k, v in predicted_args.items()
            if k in expected_args or any(value not in {"", None} for value in v)
        }
        unexpected = set(required_predicted) - set(expected_args)
        if unexpected:
            return False
    return True
