from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HANDLER_SOURCE = '''from __future__ import annotations

import re
import json
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from overrides import override


class ThesisGptOssHandler(OSSHandler):
    """Prompt-mode handler for GPT-OSS-20b via Harmony chat template."""

    reasoning_effort: str | None = None
    parallel_prompt_reminder = (
        "For parallel function-calling tasks, identify every independent "
        "function call required by the user. Return exactly one Python list "
        "containing all calls, for example [func(a=1), func(a=2)]. Do not "
        "collapse multiple requested items into one call unless the function "
        "schema explicitly asks for a list argument."
    )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]
        messages = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )
        if test_entry_id.startswith(("parallel_", "parallel_multiple_")):
            messages[0]["content"] += "\\n\\n" + self.parallel_prompt_reminder
        return {"message": [], "function": functions}

    @override
    def _format_prompt(self, messages, function):
        # Always forward reasoning_effort so Harmony uses the right compute
        # budget; the TypeError fallback drops it for tokenizers that don't
        # support the kwarg.
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": self.reasoning_effort,
        }
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _python_value(value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "True" if value else "False"
        if value is None:
            return "None"
        if isinstance(value, list):
            return "[" + ", ".join(ThesisGptOssHandler._python_value(v) for v in value) + "]"
        if isinstance(value, dict):
            return "{" + ", ".join(
                f"{ThesisGptOssHandler._python_value(k)}: {ThesisGptOssHandler._python_value(v)}"
                for k, v in value.items()
            ) + "}"
        return repr(value)

    @staticmethod
    def _balanced_json(text: str, start: int) -> str | None:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
        return None

    @staticmethod
    def _call_from_json(function_name: str, payload: str) -> str | None:
        try:
            args = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        rendered = ", ".join(
            f"{key}={ThesisGptOssHandler._python_value(value)}"
            for key, value in args.items()
        )
        return f"[{function_name}({rendered})]"

    @staticmethod
    def _clean_response(text: str) -> str:
        # Prefer the final channel if Harmony tags are emitted into the text.
        for pattern in (
            r"<\\|channel\\|>final<\\|message\\|>(.*)",
            r"<\\|start\\|>assistant<\\|channel\\|>final<\\|message\\|>(.*)",
        ):
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                text = match.group(1)
                break
        text = re.sub(r"<\\|[^>]+\\|>", "", text)
        text = text.strip()

        # With skip_special_tokens=True, Harmony markers may be decoded as plain
        # words, e.g. "assistantcommentary to=functions.foo json{...}".
        tool_matches = list(
            re.finditer(
                r"to=functions\\.([A-Za-z_][\\w]*(?:\\.[A-Za-z_][\\w]*)*)\\s+json\\s*\\{",
                text,
            )
        )
        if tool_matches:
            match = tool_matches[-1]
            payload = ThesisGptOssHandler._balanced_json(text, match.end() - 1)
            if payload:
                call = ThesisGptOssHandler._call_from_json(match.group(1), payload)
                if call:
                    return call

        # High-reasoning outputs often include stripped "assistantfinal" text.
        for marker in ("assistantfinal", "final"):
            idx = text.rfind(marker)
            if idx != -1:
                tail = text[idx + len(marker) :].strip()
                bracket = tail.find("[")
                if bracket != -1:
                    tail = tail[bracket:]
                    if tail.count("[") > tail.count("]"):
                        tail += "]"
                    end = tail.rfind("]")
                    if end != -1:
                        return tail[: end + 1].strip()

        calls = re.findall(r"\\[[A-Za-z_][\\w\\.]*\\([^\\n\\]]*\\)\\]", text)
        if calls:
            return calls[-1].strip()

        return text

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_response = api_response.choices[0].text
        cleaned_response = self._clean_response(raw_response)
        return {
            "model_responses": cleaned_response,
            "raw_model_responses": raw_response,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(result, has_tool_call_tag)


class ThesisGptOssLowHandler(ThesisGptOssHandler):
    reasoning_effort = "low"


class ThesisGptOssHighHandler(ThesisGptOssHandler):
    reasoning_effort = "high"


# ── FC (native function-calling) variants ─────────────────────────────────

class ThesisGptOssFCHandler(ThesisGptOssHandler):
    """FC-mode: functions forwarded as tools= in the Harmony chat template."""

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @override
    def _format_prompt(self, messages, function):
        tools = [{"type": "function", "function": f} for f in function] if function else None
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": self.reasoning_effort,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _extract_harmony_tool_calls(self, text: str) -> list | None:
        for pattern in (
            r"<\\|channel\\|>final<\\|message\\|>(.*)",
            r"<\\|start\\|>assistant<\\|channel\\|>final<\\|message\\|>(.*)",
        ):
            m = re.search(pattern, text, flags=re.DOTALL)
            if m:
                text = m.group(1)
                break
        text = re.sub(r"<\\|[^>]+\\|>", "", text)
        tool_matches = list(
            re.finditer(
                r"to=functions\\.([A-Za-z_][\\w]*(?:\\.[A-Za-z_][\\w]*)*)\\s+json\\s*\\{",
                text,
            )
        )
        if not tool_matches:
            return None
        calls = []
        for m in tool_matches:
            payload = self._balanced_json(text, m.end() - 1)
            if payload:
                calls.append({m.group(1): payload})
        return calls if calls else None

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_response = api_response.choices[0].text
        tool_calls = self._extract_harmony_tool_calls(raw_response)
        model_responses = tool_calls if tool_calls is not None else self._clean_response(raw_response)
        return {
            "model_responses": model_responses,
            "raw_model_responses": raw_response,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        content = model_response_data["raw_model_responses"]
        inference_data["message"].append({"role": "assistant", "content": content})
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if not isinstance(result, list):
            raise ValueError(f"FC decode_ast expected list, got {type(result)}: {result!r}")
        decoded = []
        for item in result:
            for func_name, args in item.items():
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                decoded.append({func_name: args})
        return decoded

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(result)


class ThesisGptOssLowFCHandler(ThesisGptOssFCHandler):
    reasoning_effort = "low"


class ThesisGptOssHighFCHandler(ThesisGptOssFCHandler):
    reasoning_effort = "high"
'''


IMPORT_LINE = (
    "from bfcl_eval.model_handler.local_inference.thesis_gpt_oss import (\n"
    "    ThesisGptOssHighFCHandler,\n"
    "    ThesisGptOssHighHandler,\n"
    "    ThesisGptOssLowFCHandler,\n"
    "    ThesisGptOssLowHandler,\n"
    ")\n"
)

QWEN_IMPORT_LINE = (
    "from bfcl_eval.model_handler.local_inference.qwen import "
    "Qwen3MatchedSamplingHandler, QwenChatTemplateHandler, "
    "QwenChatTemplateNoThinkingHandler, QwenHandler\n"
)

QWEN_CHAT_TEMPLATE_CLASS = '''


class QwenChatTemplateHandler(QwenHandler):
    @override
    def _format_prompt(self, messages, function):
        """
        Experimental prompt-mode Qwen handler.

        BFCL has already inserted its AST-format function-calling instructions
        and function docs into the messages. Do not pass `tools=...` here:
        that would switch Qwen into native tool-call formatting, while the
        BFCL AST evaluator expects plain Python-style calls.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class QwenChatTemplateNoThinkingHandler(QwenHandler):
    @override
    def _format_prompt(self, messages, function):
        """
        Experimental prompt-mode Qwen handler with native template and
        thinking disabled. This keeps BFCL's AST prompt-mode instructions but
        asks Qwen's template to emit the empty think block before the answer.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


class Qwen3MatchedSamplingHandler(QwenHandler):
    """QwenHandler prompt formatting with Qwen3's effective sampling config."""

    top_p = 0.95
    top_k = 20
'''


MODEL_ENTRIES = '''    # Thesis cross-task transfer models: exact local checkpoints.
    "Qwen/Qwen3-0.6B": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        display_name="Qwen3-0.6B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3-0.6B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-1.7B": ModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        display_name="Qwen3-1.7B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3-1.7B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-4B": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        display_name="Qwen3-4B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3-4B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-8B": ModelConfig(
        model_name="Qwen/Qwen3-8B",
        display_name="Qwen3-8B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3-8B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3.5-4B": ModelConfig(
        model_name="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5-4B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3.5-4B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3.5-9B": ModelConfig(
        model_name="Qwen/Qwen3.5-9B",
        display_name="Qwen3.5-9B (Prompt) (Local)",
        url="https://huggingface.co/Qwen/Qwen3.5-9B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b_low": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b low reasoning (Prompt) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssLowHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b_high": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b high reasoning (Prompt) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssHighHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b_low-FC": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b low reasoning (FC) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssLowFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b_high-FC": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b high reasoning (FC) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssHighFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    # BFCL evaluation unescapes result folder names by replacing "_" with "/".
    # These aliases let files generated as openai_gpt-oss-20b_{low,high}
    # evaluate through the official runner without changing generation names.
    "openai/gpt-oss-20b/low": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b low reasoning (Prompt) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssLowHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b/high": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="gpt-oss-20b high reasoning (Prompt) (Local)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=ThesisGptOssHighHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''


def patch_model_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    qwen_import = "from bfcl_eval.model_handler.local_inference.qwen import QwenHandler\n"
    if qwen_import in text:
        text = text.replace(qwen_import, QWEN_IMPORT_LINE)

    if "ThesisGptOssHighFCHandler" not in text:
        import_anchor = "from bfcl_eval.model_handler.local_inference.think_agent import ThinkAgentHandler\n"
        if import_anchor not in text:
            raise RuntimeError(f"Could not find import anchor in {path}")
        text = text.replace(import_anchor, import_anchor + IMPORT_LINE)

    if '"openai/gpt-oss-20b_low"' not in text:
        map_anchor = "local_inference_model_map = {\n"
        if map_anchor not in text:
            raise RuntimeError(f"Could not find local_inference_model_map in {path}")
        text = text.replace(map_anchor, map_anchor + MODEL_ENTRIES, 1)
    else:
        # File already has the original thesis entries; idempotently add any
        # models that were introduced in later versions of MODEL_ENTRIES.
        new_models = {
            '"Qwen/Qwen3-0.6B"': '    "Qwen/Qwen3-0.6B": ModelConfig(\n'
                '        model_name="Qwen/Qwen3-0.6B",\n'
                '        display_name="Qwen3-0.6B (Prompt) (Local)",\n'
                '        url="https://huggingface.co/Qwen/Qwen3-0.6B",\n'
                '        org="Qwen",\n'
                '        license="apache-2.0",\n'
                '        model_handler=QwenHandler,\n'
                '        input_price=None,\n'
                '        output_price=None,\n'
                '        is_fc_model=False,\n'
                '        underscore_to_dot=False,\n'
                '    ),\n',
            '"Qwen/Qwen3-1.7B"': '    "Qwen/Qwen3-1.7B": ModelConfig(\n'
                '        model_name="Qwen/Qwen3-1.7B",\n'
                '        display_name="Qwen3-1.7B (Prompt) (Local)",\n'
                '        url="https://huggingface.co/Qwen/Qwen3-1.7B",\n'
                '        org="Qwen",\n'
                '        license="apache-2.0",\n'
                '        model_handler=QwenHandler,\n'
                '        input_price=None,\n'
                '        output_price=None,\n'
                '        is_fc_model=False,\n'
                '        underscore_to_dot=False,\n'
                '    ),\n',
            '"Qwen/Qwen3-8B"': '    "Qwen/Qwen3-8B": ModelConfig(\n'
                '        model_name="Qwen/Qwen3-8B",\n'
                '        display_name="Qwen3-8B (Prompt) (Local)",\n'
                '        url="https://huggingface.co/Qwen/Qwen3-8B",\n'
                '        org="Qwen",\n'
                '        license="apache-2.0",\n'
                '        model_handler=QwenHandler,\n'
                '        input_price=None,\n'
                '        output_price=None,\n'
                '        is_fc_model=False,\n'
                '        underscore_to_dot=False,\n'
                '    ),\n',
        }
        anchor = '"Qwen/Qwen3-4B": ModelConfig(\n'
        for key, entry in new_models.items():
            if key not in text:
                text = text.replace(anchor, entry + anchor, 1)

    path.write_text(text, encoding="utf-8")


def patch_qwen_handler(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "class QwenChatTemplateHandler" not in text:
        text = text.rstrip() + QWEN_CHAT_TEMPLATE_CLASS
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register thesis Qwen3.5 and GPT-OSS models in an editable BFCL checkout."
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        required=True,
        help="Path to gorilla/berkeley-function-call-leaderboard.",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = args.bfcl_root.resolve()
    package_dir = root / "bfcl_eval"
    model_config = package_dir / "constants" / "model_config.py"
    handler_path = package_dir / "model_handler" / "local_inference" / "thesis_gpt_oss.py"
    qwen_handler_path = package_dir / "model_handler" / "local_inference" / "qwen.py"
    if not model_config.exists():
        raise FileNotFoundError(f"BFCL model_config.py not found at {model_config}")

    if not args.no_backup:
        backup = model_config.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(model_config, backup)

    handler_path.write_text(HANDLER_SOURCE, encoding="utf-8")
    patch_qwen_handler(qwen_handler_path)
    patch_model_config(model_config)
    print(f"Wrote {handler_path}")
    print(f"Patched {qwen_handler_path}")
    print(f"Patched {model_config}")
    print(
        "Registered: Qwen/Qwen3-{0.6B,1.7B,4B,8B}, Qwen/Qwen3.5-{0.8B,2B,4B,9B}, "
        "Qwen/Qwen3.5-{4B,9B}-chat-template, "
        "Qwen/Qwen3.5-{4B,9B}-chat-template-no-thinking, "
        "openai/gpt-oss-20b_low, openai/gpt-oss-20b_high"
    )


if __name__ == "__main__":
    main()
