from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.bfcl import load_processed_bfcl_examples


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DatasetLoader(ABC):
	"""Common interface for dataset loaders in the probing pipeline.

	Every loader must expose ``load_probe_fields`` which returns the triplet
	(prompts, labels, [answers]) consumed by extraction and probing scripts.
	"""

	name: str  # short id used in output paths, e.g. "bfcl"

	@abstractmethod
	def load_probe_fields(
		self,
		split: str = "train",
		max_examples: int | None = None,
		return_answers: bool = False,
		answer_source: str = "generated_solutions",
		generated_answer_index: int | None = None,
	) -> tuple[list[str], np.ndarray] | tuple[list[str], np.ndarray, list]:
		"""Return ``(prompts, labels)`` or ``(prompts, labels, answers)``.

		Parameters
		----------
		split : str
			Dataset split (e.g. "train", "validation", "test", "eval").
		max_examples : int | None
			Cap on the number of examples to load.
		return_answers : bool
			If True, also return generated solution texts.
		answer_source : str
			Which field to use for answers. Default "generated_solutions".
		generated_answer_index : int | None
			Select a single rollout (e.g. -1 for last). None = all rollouts.
		"""
		...


# ---------------------------------------------------------------------------
# BFCL (English-only function-calling transfer experiment)
# ---------------------------------------------------------------------------


@dataclass
class BFCLLoader(DatasetLoader):
	"""Load processed BFCL examples for English-only cross-task probing.

	The raw BFCL files are prepared with ``src.bfcl.prepare_bfcl_data`` into a
	JSONL file containing stable ids, prompts, function definitions, references,
	categories, and deterministic splits. If ``success_path`` is provided,
	labels are read from an evaluated per-example CSV; otherwise labels default
	to zero so prompt-only stages can still reuse the loader.
	"""

	name: str = "bfcl"

	data_path: str = "data/bfcl/processed_bfcl.jsonl"
	success_path: str | None = None
	tokenizer: object | None = None
	enable_thinking: bool = False
	reasoning_effort: str | None = None

	def _load_success(self) -> dict[str, float]:
		if self.success_path is None:
			return {}
		path = Path(self.success_path)
		if not path.exists():
			raise FileNotFoundError(f"BFCL success CSV not found: {path}")
		df = pd.read_csv(path)
		if "example_id" not in df.columns or "empirical_success" not in df.columns:
			raise ValueError(
				f"BFCL success CSV must contain example_id and empirical_success: {path}"
			)
		return {
			str(row.example_id): float(row.empirical_success)
			for row in df.itertuples(index=False)
		}

	def load_probe_fields(
		self,
		split: str = "train",
		max_examples: int | None = None,
		return_answers: bool = False,
		answer_source: str = "ground_truth",
		generated_answer_index: int | None = None,
	):
		from src.bfcl import format_bfcl_chat_prompt

		rows = load_processed_bfcl_examples(self.data_path, split=split, max_examples=max_examples)
		success = self._load_success()
		prompts = [
			format_bfcl_chat_prompt(
				row,
				self.tokenizer,
				enable_thinking=self.enable_thinking,
				reasoning_effort=self.reasoning_effort,
			)
			for row in rows
		]
		labels = np.asarray(
			[success.get(str(row["example_id"]), 0.0) for row in rows],
			dtype=np.float32,
		)
		if return_answers:
			if answer_source != "ground_truth":
				raise ValueError("BFCLLoader currently supports answer_source='ground_truth' only.")
			return prompts, labels, [row.get("ground_truth") for row in rows]
		return prompts, labels


# ---------------------------------------------------------------------------
# Per-file success rate loader
# ---------------------------------------------------------------------------

_KNOWN_LANGUAGES = {
    "English", "Chinese", "Italian", "French", "Swahili",
    "Russian", "Turkish", "Arabic", "Thai", "Telugu",
}

_SUCCESS_RATE_COLUMNS = [
    "problem_id", "base_problem_id", "language", "model",
    "success_rate", "n_rollouts", "n_correct", "has_truncated_rollouts",
]


def _parse_model_language(stem: str) -> tuple[str, str] | None:
    """Parse '{Model}_{Language}' stem into (model_tag, language).

    Handles model tags containing underscores by matching against a known
    set of language names from the right side of the stem.
    """
    parts = stem.split("_")
    for split_at in range(len(parts) - 1, 0, -1):
        language = "_".join(parts[split_at:])
        model_tag = "_".join(parts[:split_at])
        if language in _KNOWN_LANGUAGES:
            return model_tag, language
    return None


def load_success_rates(
    directory: Path | str,
    models: list[str] | set[str] | None = None,
    languages: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Load per-(model, language) success rate CSVs from *directory*.

    Each file must be named ``{Model}_{Language}.csv`` and contain the
    standard 8-column schema.  Files that cannot be parsed are silently
    skipped with a warning.

    Parameters
    ----------
    directory:
        Path to ``outputs/success_rates/MATH/`` (or equivalent).
    models:
        Optional allowlist of model tags (e.g. ``["Qwen3-4B", "Qwen3-8B"]``).
    languages:
        Optional allowlist of language names (e.g. ``["English", "French"]``).

    Returns
    -------
    pd.DataFrame with columns defined by ``_SUCCESS_RATE_COLUMNS``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Success-rate directory not found: {directory}")

    model_filter = set(models) if models is not None else None
    language_filter = set(languages) if languages is not None else None

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(directory.glob("*.csv")):
        parsed = _parse_model_language(csv_path.stem)
        if parsed is None:
            import warnings
            warnings.warn(f"load_success_rates: skipping unrecognised file {csv_path.name}")
            continue
        model_tag, language = parsed
        if model_filter is not None and model_tag not in model_filter:
            continue
        if language_filter is not None and language not in language_filter:
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No matching success-rate CSVs found in {directory} "
            f"(models={models}, languages={languages})."
        )

    result = pd.concat(frames, ignore_index=True)
    for col in ["problem_id", "base_problem_id", "language", "model"]:
        result[col] = result[col].astype(str)
    result["success_rate"] = pd.to_numeric(result["success_rate"], errors="coerce")
    return result


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
	"bfcl": BFCLLoader,
}


def get_loader(dataset: str, **kwargs) -> DatasetLoader:
	"""Instantiate a dataset loader by name.

	Parameters
	----------
	dataset : str
		Key in ``DATASET_REGISTRY`` (e.g. "bfcl").
	**kwargs
		Passed to the loader constructor.
	"""
	cls = DATASET_REGISTRY.get(dataset)
	if cls is None:
		raise ValueError(
			f"Unknown dataset: {dataset!r}. "
			f"Available: {sorted(DATASET_REGISTRY)}"
		)
	return cls(**kwargs)
