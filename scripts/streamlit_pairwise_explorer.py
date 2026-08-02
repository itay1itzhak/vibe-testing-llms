#!/usr/bin/env python3
"""
Streamlit UI for exploring Stage 5b pairwise comparisons.

This app scans a results directory for `pairwise-comparison_*.json` artifacts
written by `scripts/stage_5b_pairwise_comparison.py`, then lets you:
- switch between discovered configurations (persona/gen/filter/prompt_type)
- select a model pair
- navigate samples one-by-one
- inspect prompt, both model outputs, and each judge's full judgments
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Third-party used for the All-samples tab tables.
import pandas as pd

# Reduce noisy NumExpr warnings in Streamlit logs (safe default).
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import streamlit as st

# Ensure repository root is importable when running via `streamlit run ...`.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vibe_testing.ui.pairwise_explorer_io import (  # noqa: E402
    ModelPairKey,
    PairwiseConfigKey,
    build_task_id_union,
    discover_judges_for_pair,
    discover_model_pairs,
    discover_pairwise_configs,
    format_prompt_type,
    index_records_by_task_id,
    list_pairwise_artifacts_for_pair_and_judge,
    load_pairwise_json,
    select_latest_pairwise_artifact,
    stage6_display_winner_name_from_row,
    stage6_sample_id_from_row,
    win_counts_by_model_name,
)
from src.vibe_testing.ui.pairwise_explorer_stats import (  # noqa: E402
    align_pairwise_judgment_types_to_human_scope,
    build_long_winner_table_with_meta,
    build_overall_winner_table,
    build_overall_winner_table_with_meta,
    compute_dimension_judge_pair_agreement,
    compute_judge_pair_agreement,
    compute_judge_pair_agreement_by_group,
    filter_agreement_df_between_judge_groups,
    filter_agreement_df_to_within_judge_group,
    filter_item_ids_by_human_confidence,
    filter_item_ids_to_human_annotated,
    is_human_judge_token,
    is_eval_or_parse_error_rationale,
    recompute_overall_from_dimensions_with_controls,
    recompute_overall_from_dimensions_with_controls_weighted,
    split_judges_by_group,
    summarize_dimension_agreement_mean_std,
    summarize_dimension_human_vs_llm_means,
    summarize_human_vs_llm_agreement_means,
)
from src.vibe_testing.ui.artifact_selection import (  # noqa: E402
    PairwiseArtifactSelectionKey,
    choose_stage6_default_artifact,
    format_artifact_option_label,
    get_saved_selected_path,
    load_pairwise_artifact_selections,
    save_pairwise_artifact_selections,
    selection_file_for_run_dir,
    upsert_selection_record,
)
from src.vibe_testing.analysis.io import (
    AnalysisInputLoader,
    PAIRWISE_DIMENSIONS,
)  # noqa: E402
from src.vibe_testing.pairwise_artifact_diagnostics import (  # noqa: E402
    PairwiseArtifactLoadContext,
    PairwiseArtifactLoadError,
    wrap_pairwise_artifact_load_error,
)
from src.vibe_testing.analysis.dimension_omits import (  # noqa: E402
    apply_pairwise_dimension_omits,
    normalize_omit_dimensions,
    recompute_pairwise_overall_winner,
    recompute_pairwise_overall_winner_dimension_weighted,
)
from src.vibe_testing.analysis.persona_pairwise_weights import (  # noqa: E402
    load_persona_pairwise_dimension_weights,
)
from src.vibe_testing.analysis.judge_agreement import (  # noqa: E402
    compute_judge_agreement_for_joint_preference,
)
from src.vibe_testing.analysis.joint_preference import (  # noqa: E402
    compute_joint_preference_matrices,
    compute_joint_preference_long_by_judge,
    compute_cluster_aware_paired_tests_for_joint_preference,
)
from src.vibe_testing.analysis.pairwise import (  # noqa: E402
    _build_objective_lookup,  # noqa: SLF001
    compute_dimension_win_rates,
)
from src.vibe_testing.ui.stage6_backbone import (  # noqa: E402
    build_stage6_outcomes_by_judge_token,
)
from src.vibe_testing.ui.persona_weight_selection import (  # noqa: E402
    load_persona_weight_config_selection,
    save_persona_weight_config_selection,
    selection_file_for_run_dir as persona_weight_selection_file_for_run_dir,
)
from src.vibe_testing.ui.aggregate_filter_selection import (  # noqa: E402
    AggregateFilterSelection,
    load_aggregate_filter_selection,
    save_aggregate_filter_selection,
    selection_file_for_run_dir as aggregate_filter_selection_file_for_run_dir,
)
from src.vibe_testing.utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

RUNS_ROOT_DIR = project_root / "runs" / "experiments"
DEFAULT_RUNS_SUBDIR = "new_gpt5_100_samples"
PERSONA_COMPONENTS_CONFIG_PATH = (
    project_root / "configs" / "experiments" / "components" / "personas.yaml"
)


def _artifact_status_row(
    *,
    persona: str,
    generator_model: str,
    filter_model: str,
    prompt_type: str,
    pairwise_judgment_type: str,
    model_pair: str,
    judge_token: str,
    artifact_path: str,
    reason: str,
    stage6_default_path: str,
    load_status: str,
    load_error: str = "",
) -> Dict[str, Any]:
    """Build a bounded artifact status row for aggregate diagnostics."""

    return {
        "persona": str(persona),
        "generator_model": str(generator_model),
        "filter_model": str(filter_model),
        "prompt_type": str(prompt_type),
        "pairwise_judgment_type": str(pairwise_judgment_type),
        "model_pair": str(model_pair),
        "judge_token": str(judge_token),
        "artifact_path": str(artifact_path),
        "reason": str(reason),
        "stage6_default_path": str(stage6_default_path),
        "load_status": str(load_status),
        "load_error": str(load_error or ""),
    }


def _emit_pairwise_load_failure_to_ui(
    *,
    exc: Exception,
    prefix: str,
    stop_after: bool,
) -> None:
    """Render a structured pairwise load failure in the Streamlit UI."""

    st.error(f"{prefix}: {exc}")
    if isinstance(exc, PairwiseArtifactLoadError):
        with st.expander("Pairwise load failure details", expanded=False):
            st.json(exc.context.as_dict())
    if stop_after:
        st.stop()


def _normalize_widget_choice(
    current_value: Optional[str],
    *,
    allowed_values: Sequence[str],
    desired_default: str,
) -> str:
    """
    Normalize a single-choice widget value against the allowed options.

    Args:
        current_value: Existing value from session state, if any.
        allowed_values: Allowed option values in display order.
        desired_default: Default value to use when the current one is missing or
            invalid.

    Returns:
        str: Valid widget value.
    """

    allowed = [str(x) for x in allowed_values]
    if not allowed:
        raise ValueError("allowed_values must contain at least one option.")
    if str(desired_default) not in set(allowed):
        raise ValueError(
            f"desired_default={desired_default!r} is not present in allowed_values."
        )
    if current_value is None:
        return str(desired_default)
    value = str(current_value)
    return value if value in set(allowed) else str(desired_default)


def _normalize_multiselect_state(
    current_values: Optional[Sequence[str]],
    *,
    options: Sequence[str],
    desired_default: Sequence[str],
) -> List[str]:
    """
    Normalize multiselect state against the current option list.

    Args:
        current_values: Existing value from session state or saved selections.
        options: Allowed options in display order.
        desired_default: Default selection to use when initializing or when the
            current selection contains only stale values.

    Returns:
        List[str]: Valid multiselect selection preserving the incoming order
            where possible.
    """

    option_list = [str(x) for x in options]
    option_set = set(option_list)
    default_list = [str(x) for x in desired_default if str(x) in option_set]
    fallback = default_list if default_list else list(option_list)

    if current_values is None:
        return fallback

    current_list = [str(x) for x in current_values]
    normalized = [x for x in current_list if x in option_set]
    if current_list and not normalized:
        return fallback
    return normalized


def _get_record_correctness(
    record: Dict[str, Any],
    correctness_mode: str,
    objective_lookup: Optional[Dict],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Look up base and plus pass@1 values for both models in a raw pairwise record.

    Args:
        record: Stage-5b pairwise record dict.
        correctness_mode: Current correctness handling mode.
        objective_lookup: Pre-built objective lookup, or None.

    Returns:
        Tuple of (base_a, base_b, plus_a, plus_b). All None when not applicable.
    """
    if correctness_mode == "ignore" or objective_lookup is None:
        return None, None, None, None

    from src.vibe_testing.analysis.dimension_omits import _lookup_sample_correctness

    row_series = pd.Series(record)
    return _lookup_sample_correctness(row_series, objective_lookup)


def _canonicalize_omit_dimension_selection(
    selection: Sequence[str],
) -> List[str]:
    """
    Normalize omit-dimension selections to canonical pairwise dimension keys.

    This keeps the Streamlit widget options stable and user-facing even when older
    saved selections contain legacy aliases such as ``efficiency`` or
    ``frustration``.

    Args:
        selection: Raw selected tokens from session state or saved settings.

    Returns:
        List[str]: Canonical omit dimensions in ``PAIRWISE_DIMENSIONS`` order.
    """
    pairwise_omits, _subjective_omits = normalize_omit_dimensions(selection)
    return [dim for dim in PAIRWISE_DIMENSIONS if str(dim) in pairwise_omits]


def _available_judgment_types(configs: Sequence[PairwiseConfigKey]) -> List[str]:
    """
    Return the sorted judgment types present in a config slice.

    Args:
        configs: Candidate config slice.

    Returns:
        List[str]: Sorted distinct ``pairwise_judgment_type`` values.
    """
    return sorted({str(cfg.pairwise_judgment_type) for cfg in configs}, key=str)


def _select_exact_pairwise_config(
    *,
    configs: Sequence[PairwiseConfigKey],
    prompt_label: str,
    judgment_type: str,
) -> PairwiseConfigKey:
    """
    Resolve exactly one pairwise config from the current sidebar slice.

    Args:
        configs: Candidate configs already filtered by persona/gen/filter.
        prompt_label: Selected prompt label as shown in the UI.
        judgment_type: Selected pairwise judgment type.

    Returns:
        PairwiseConfigKey: The exact matching config.

    Raises:
        ValueError: If no config or multiple configs match the requested slice.
    """
    matching = [
        cfg
        for cfg in configs
        if format_prompt_type(cfg.prompt_type) == str(prompt_label)
        and str(cfg.pairwise_judgment_type) == str(judgment_type)
    ]
    if not matching:
        raise ValueError(
            "No matching configuration found for the selected fields. "
            f"prompt_type={prompt_label!r} pairwise_judgment_type={judgment_type!r}"
        )
    if len(matching) != 1:
        raise ValueError(
            "Multiple matching configurations found for the selected fields. "
            f"prompt_type={prompt_label!r} pairwise_judgment_type={judgment_type!r} "
            f"matches={[asdict(cfg) for cfg in matching]!r}"
        )
    return matching[0]


def _filter_aggregate_pairwise_configs(
    *,
    configs: Sequence[PairwiseConfigKey],
    personas: Sequence[str],
    generator_models: Sequence[str],
    filter_models: Sequence[str],
    prompt_labels: Sequence[str],
    judgment_types: Sequence[str],
) -> List[PairwiseConfigKey]:
    """
    Filter aggregate configs across all exposed selection axes.

    Args:
        configs: Candidate config list.
        personas: Selected persona aliases.
        generator_models: Selected generator model names.
        filter_models: Selected filter model names.
        prompt_labels: Selected prompt labels.
        judgment_types: Selected pairwise judgment types.

    Returns:
        List[PairwiseConfigKey]: Matching config slice.
    """
    persona_set = {str(x) for x in personas}
    gen_set = {str(x) for x in generator_models}
    filter_set = {str(x) for x in filter_models}
    prompt_set = {str(x) for x in prompt_labels}
    judgment_set = {str(x) for x in judgment_types}
    return [
        cfg
        for cfg in configs
        if cfg.persona in persona_set
        and cfg.generator_model in gen_set
        and cfg.filter_model in filter_set
        and format_prompt_type(cfg.prompt_type) in prompt_set
        and str(cfg.pairwise_judgment_type) in judgment_set
    ]


def _record_public_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the public metadata payload for a pairwise record.

    Args:
        record: Raw Stage-5b record or Stage-6-normalized row dict.

    Returns:
        Dict[str, Any]: Public metadata dict, or an empty dict when unavailable.
    """
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _record_uses_human_judge(
    record: Dict[str, Any], judge_token: Optional[str] = None
) -> bool:
    """
    Return True when a record belongs to a human annotator judge.

    Args:
        record: Raw Stage-5b record or Stage-6-normalized row dict.
        judge_token: Optional UI judge token associated with the record.

    Returns:
        bool: True when any available judge identifier matches the human prefix.
    """
    candidates = [
        judge_token,
        record.get("_judge_token"),
        record.get("judge_model_name"),
        record.get("raw_judge_model_name"),
    ]
    return any(
        is_human_judge_token(str(candidate))
        for candidate in candidates
        if candidate is not None and str(candidate).strip()
    )


def _resolve_direct_human_overall_winner(
    record: Dict[str, Any], judge_token: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a human annotator's direct overall choice into a winner name.

    Args:
        record: Raw Stage-5b record or Stage-6-normalized row dict.
        judge_token: Optional UI judge token associated with the record.

    Returns:
        Optional[str]: ``model_a_name``, ``model_b_name``, or ``"tie"``. Returns
        None for non-human judges.

    Raises:
        ValueError: If the record is human-authored but does not include a valid
            ``human_overall_choice`` field.
    """
    if not _record_uses_human_judge(record, judge_token=judge_token):
        return None

    choice = record.get("human_overall_choice")
    if choice is None:
        choice = _record_public_metadata(record).get("human_overall_choice")
    token = str(choice or "").strip()
    if token == "A":
        return str(record.get("model_a_name", "model_a"))
    if token == "B":
        return str(record.get("model_b_name", "model_b"))
    if token == "tie":
        return "tie"
    raise ValueError(
        "Human-annotator record is missing a valid direct overall choice. "
        f"judge_token={judge_token!r} judge_model_name={record.get('judge_model_name')!r} "
        f"human_overall_choice={choice!r} task_id={record.get('task_id')!r}"
    )


def _apply_direct_human_overall_choice_to_stage6_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Override Stage-6 overall winners with direct human choices when available.

    Args:
        df: Stage-6-shaped pairwise DataFrame.

    Returns:
        pd.DataFrame: Copy with human-judge overall winner columns overridden.
    """
    if df is None or df.empty:
        return df

    rows: List[Dict[str, Any]] = []
    changed = False
    for row in df.to_dict(orient="records"):
        direct_winner = _resolve_direct_human_overall_winner(
            row, judge_token=str(row.get("_judge_token", "") or "")
        )
        if direct_winner is None:
            rows.append(row)
            continue

        row = dict(row)
        changed = True
        row["human_overall_source"] = "human_overall_choice"
        if direct_winner == "tie":
            row["overall_winner"] = None
            row["overall_winner_label"] = "tie"
        elif direct_winner == str(row.get("model_a_name", "")):
            row["overall_winner"] = direct_winner
            row["overall_winner_label"] = "model_a"
        elif direct_winner == str(row.get("model_b_name", "")):
            row["overall_winner"] = direct_winner
            row["overall_winner_label"] = "model_b"
        else:
            raise ValueError(
                "Direct human overall choice could not be mapped onto Stage-6 row models. "
                f"direct_winner={direct_winner!r} model_a_name={row.get('model_a_name')!r} "
                f"model_b_name={row.get('model_b_name')!r}"
            )
        row["pairwise_valid_for_overall"] = True
        rows.append(row)

    if not changed:
        return df
    return pd.DataFrame(rows)


def _load_persona_weight_catalog() -> Dict[str, Any]:
    """
    Load the experiment persona catalog used for Streamlit weight selection.

    The experiment components file is the source of truth for which persona aliases
    map to which YAML configs. This avoids surfacing deprecated legacy YAMLs in the
    weight-selection UI.

    Returns:
        Dict[str, Any]: Catalog containing alias/config/user-id mappings.
    """
    components_path = PERSONA_COMPONENTS_CONFIG_PATH.expanduser().resolve()
    payload = load_config(str(components_path))
    personas = payload.get("personas", {})
    if not isinstance(personas, dict) or not personas:
        raise ValueError(
            "Persona components config is missing a non-empty 'personas' mapping: "
            f"{str(components_path)!r}"
        )

    alias_to_user_id: Dict[str, str] = {}
    alias_to_config_path: Dict[str, str] = {}
    config_path_to_aliases: Dict[str, List[str]] = {}
    config_path_to_user_id: Dict[str, str] = {}
    ordered_config_paths: List[str] = []
    seen_paths: set[str] = set()

    for alias, spec in sorted(personas.items(), key=lambda kv: str(kv[0])):
        if not isinstance(spec, dict):
            continue
        config_ref = spec.get("config")
        if not config_ref or not str(config_ref).strip():
            continue

        config_path = Path(str(config_ref).strip())
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        else:
            config_path = config_path.expanduser().resolve()

        if not config_path.exists():
            raise FileNotFoundError(
                "Persona config referenced by experiment components file does not exist: "
                f"alias={alias!r} path={str(config_path)!r}"
            )

        persona_cfg = load_config(str(config_path))
        user_id = str(persona_cfg.get("user_id") or "").strip()
        if not user_id:
            raise ValueError(
                "Persona config referenced by experiment components file is missing "
                f"'user_id': alias={alias!r} path={str(config_path)!r}"
            )

        alias_key = str(alias).strip()
        config_path_str = str(config_path)
        alias_to_user_id[alias_key] = user_id
        alias_to_config_path[alias_key] = config_path_str
        config_path_to_aliases.setdefault(config_path_str, []).append(alias_key)
        config_path_to_user_id[config_path_str] = user_id
        if config_path_str not in seen_paths:
            seen_paths.add(config_path_str)
            ordered_config_paths.append(config_path_str)

    return {
        "alias_to_user_id": alias_to_user_id,
        "alias_to_config_path": alias_to_config_path,
        "config_path_to_aliases": config_path_to_aliases,
        "config_path_to_user_id": config_path_to_user_id,
        "config_paths": ordered_config_paths,
    }


def _resolve_weight_user_id(
    *,
    persona_alias: str,
    raw_user_id: object,
    alias_to_user_id: Dict[str, str],
) -> str:
    """
    Resolve the canonical user_id used for persona-weight lookups.

    Args:
        persona_alias: Experiment/run persona alias (e.g. ``novice_user``).
        raw_user_id: User id already present on the record, if any.
        alias_to_user_id: Alias -> canonical user_id mapping from the components file.

    Returns:
        str: Canonical user id when resolvable, otherwise the existing non-empty id.

    Raises:
        ValueError: If only a persona alias is available but it cannot be resolved.
    """
    alias = str(persona_alias or "").strip()
    user_id = str(raw_user_id or "").strip()

    # Preserve canonical ids already present on records.
    if user_id and user_id != alias:
        return user_id

    resolved = alias_to_user_id.get(alias)
    if resolved:
        return resolved

    if user_id:
        return user_id

    raise ValueError(
        "Could not resolve a canonical persona user_id for weighted analysis. "
        f"persona_alias={alias!r}. Check {str(PERSONA_COMPONENTS_CONFIG_PATH)!r}."
    )


def _apply_canonical_user_id_to_record(
    *,
    record: Dict[str, Any],
    persona_alias: str,
    alias_to_user_id: Dict[str, str],
) -> Dict[str, Any]:
    """
    Return a shallow copy of a pairwise record with canonicalized ``user_id``.

    Args:
        record: Raw pairwise record dict.
        persona_alias: Persona alias for the enclosing result slice.
        alias_to_user_id: Alias -> canonical user_id mapping.

    Returns:
        Dict[str, Any]: Updated record copy.
    """
    updated = dict(record)
    updated.setdefault("source_persona", str(persona_alias or "").strip())
    updated["user_id"] = _resolve_weight_user_id(
        persona_alias=persona_alias,
        raw_user_id=updated.get("user_id"),
        alias_to_user_id=alias_to_user_id,
    )
    return updated


def _canonicalize_artifact_records_for_weight_lookup(
    *,
    records: Sequence[Dict[str, Any]],
    artifact_path_str: str,
    base_dir_str: str,
) -> List[Dict[str, Any]]:
    """
    Canonicalize raw artifact records so weighted analysis can resolve persona IDs.

    Args:
        records: Raw Stage-5b pairwise records.
        artifact_path_str: Absolute artifact path.
        base_dir_str: Base results directory used to infer the persona alias.

    Returns:
        List[Dict[str, Any]]: Record copies with canonicalized ``user_id`` values.
    """
    persona_alias = _infer_persona_alias_from_artifact_path(artifact_path_str, base_dir_str)
    alias_to_user_id = _load_persona_weight_catalog()["alias_to_user_id"]
    canonical_records = [
        _apply_canonical_user_id_to_record(
            record=record,
            persona_alias=persona_alias,
            alias_to_user_id=alias_to_user_id,
        )
        for record in records
    ]
    raw_user_ids = sorted(
        {
            str(record.get("user_id", "")).strip()
            for record in records
            if str(record.get("user_id", "")).strip()
        }
    )
    canonical_user_ids = sorted(
        {
            str(record.get("user_id", "")).strip()
            for record in canonical_records
            if str(record.get("user_id", "")).strip()
        }
    )
    logger.info(
        "canonicalized artifact records for weight lookup: artifact=%s persona_alias=%s raw_user_ids=%s canonical_user_ids=%s",
        artifact_path_str,
        persona_alias,
        raw_user_ids,
        canonical_user_ids,
    )
    return canonical_records


def _infer_persona_alias_from_artifact_path(
    artifact_path_str: str, base_dir_str: str
) -> str:
    """
    Infer the persona directory token from a Stage-5b artifact path.

    Args:
        artifact_path_str: Absolute artifact path.
        base_dir_str: Base results directory.

    Returns:
        str: Persona alias from the first path segment under the base directory,
            or an empty string when it cannot be inferred.
    """
    artifact_path = Path(str(artifact_path_str)).expanduser().resolve()
    base_dir = Path(str(base_dir_str)).expanduser().resolve()
    try:
        rel = artifact_path.relative_to(base_dir)
    except Exception:
        return ""
    return str(rel.parts[0]).strip() if rel.parts else ""


def _width_kwargs(callable_obj: object, mode: str) -> Dict[str, Any]:
    """
    Return kwargs that control widget width in a Streamlit-version-safe way.

    Streamlit is migrating from `use_container_width=True/False` to
    `width="stretch"/"content"`. We support both, based on runtime function
    signatures, to avoid warnings while remaining compatible with older versions.

    Args:
        callable_obj: Streamlit function or bound method.
        mode: Either 'stretch' or 'content'.

    Returns:
        Dict[str, Any]: Keyword args to pass to the widget call.
    """

    mode_token = str(mode or "stretch").strip().lower()
    if mode_token not in {"stretch", "content"}:
        mode_token = "stretch"

    try:
        params = inspect.signature(callable_obj).parameters
    except Exception:  # noqa: BLE001
        return {"use_container_width": mode_token == "stretch"}

    if "width" in params:
        return {"width": mode_token}
    if "use_container_width" in params:
        return {"use_container_width": mode_token == "stretch"}
    return {}


def _dataframe_supports_row_selection() -> bool:
    """
    Return True if this Streamlit version supports row selection in st.dataframe.

    Returns:
        bool: True if st.dataframe exposes on_select + selection_mode parameters.
    """

    try:
        params = inspect.signature(st.dataframe).parameters
    except Exception:  # noqa: BLE001
        return False
    return "on_select" in params and "selection_mode" in params


def _ui_width_mode() -> str:
    """
    Resolve current UI width mode from session state.

    Returns:
        str: 'content' for compact mode, else 'stretch'.
    """

    density = str(st.session_state.get("layout_density", "comfortable")).strip().lower()
    return "content" if density == "compact" else "stretch"


def _is_stage6_tie_breaker(mode: str) -> bool:
    """
    Return True if the selected tie breaker should match Stage 6 semantics.

    Args:
        mode: UI tie-breaker token.

    Returns:
        bool: True if the token starts with 'stage6_'.
    """

    return str(mode or "").strip().lower().startswith("stage6_")


def _stage6_tie_breaker_mode(mode: str) -> str:
    """
    Map UI tie-breaker selection to Stage-6 tie breaker mode.

    Args:
        mode: UI tie-breaker token.

    Returns:
        str: One of {'strict','finegrained'}.
    """

    token = str(mode or "stage6_strict").strip().lower()
    if token == "stage6_finegrained":
        return "finegrained"
    return "strict"


def _dimension_tie_breaker_mode(mode: str) -> str:
    """
    Map UI tie-breaker selection to Sample-view dimension recomputation mode.

    Stage-6 strict corresponds to using stored winners (no swap-agreement tying).

    Args:
        mode: UI tie-breaker token.

    Returns:
        str: One of {'stored','strict','finegrained'} (Streamlit Sample-view internal).
    """

    token = str(mode or "stored").strip().lower()
    if token in {"stage6_strict", "stored"}:
        return "stored"
    if token == "stage6_finegrained":
        return "finegrained"
    if token == "swap_agreement_strict":
        return "strict"
    if token == "strict":
        return "strict"
    if token == "finegrained":
        return "finegrained"
    return "stored"


def _default_base_dir_candidates() -> List[Path]:
    """
    Return default base directory candidates relative to current working directory.

    Returns:
        List[Path]: Candidates in priority order.
    """

    cwd = Path.cwd()
    return [cwd / "runs", cwd / "Runs"]


def _list_immediate_subdirs(root: Path) -> List[str]:
    """
    List immediate child directories under a root directory.

    Args:
        root: Root directory to list.

    Returns:
        Sorted list of directory names.
    """

    if not root.exists() or not root.is_dir():
        return []
    names: List[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            names.append(entry.name)
    return names


def _format_config_label(cfg: PairwiseConfigKey) -> str:
    """
    Render a human-friendly config label for selection.

    Args:
        cfg: PairwiseConfigKey.

    Returns:
        str: Display label.
    """

    return (
        f"persona={cfg.persona} | "
        f"gen={cfg.generator_model} | "
        f"filter={cfg.filter_model} | "
        f"prompt={format_prompt_type(cfg.prompt_type)} | "
        f"judgment={cfg.pairwise_judgment_type}"
    )


@st.cache_data(show_spinner=False)
def _cached_discover_configs(base_dir_str: str) -> List[Dict[str, Any]]:
    """
    Cached wrapper around config discovery.

    Args:
        base_dir_str: Base results directory as string.

    Returns:
        List of configs as plain dicts (Streamlit cache-friendly).
    """

    base_dir = Path(base_dir_str).expanduser().resolve()
    configs = discover_pairwise_configs(base_dir)
    return [asdict(c) for c in configs]


@st.cache_data(show_spinner=False)
def _cached_discover_pairs(
    base_dir_str: str, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Cached wrapper around model pair discovery.

    Args:
        base_dir_str: Base results directory as string.
        cfg: Config dict compatible with PairwiseConfigKey.

    Returns:
        List of pairs as dicts.
    """

    base_dir = Path(base_dir_str).expanduser().resolve()
    config = PairwiseConfigKey(**cfg)
    pairs = discover_model_pairs(base_dir, config)
    return [asdict(p) for p in pairs]


@st.cache_data(show_spinner=False)
def _cached_load_latest_records(
    base_dir_str: str,
    cfg: Dict[str, Any],
    pair_dict: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, str]]:
    """
    Load the latest pairwise artifact for each judge and index by task_id.

    Args:
        base_dir_str: Base results directory as string.
        cfg: Config dict.
        pair_dict: Model pair dict.

    Returns:
        Tuple of:
        - task_ids: sorted union of task_ids across judges
        - index_by_judge: judge -> task_id -> record
        - artifact_by_judge: judge -> artifact path string
    """

    base_dir = Path(base_dir_str).expanduser().resolve()
    config = PairwiseConfigKey(**cfg)
    pair = ModelPairKey(**pair_dict)
    persona_weight_catalog = _load_persona_weight_catalog()
    alias_to_user_id = persona_weight_catalog["alias_to_user_id"]

    judges = discover_judges_for_pair(base_dir, config, pair)
    if not judges:
        raise ValueError(
            "No judge outputs found for the selected configuration and model pair."
        )

    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]] = {}
    artifact_by_judge: Dict[str, str] = {}
    attempted_count = 0
    loaded_count = 0
    failure_messages: List[str] = []

    for judge in judges:
        artifacts = list_pairwise_artifacts_for_pair_and_judge(
            base_dir, config, pair, judge
        )
        if not artifacts:
            logger.debug(
                "Latest-record loader found no artifacts for judge: persona=%s pair=%s judge=%s",
                config.persona,
                pair.label,
                judge,
            )
            continue
        latest = select_latest_pairwise_artifact(artifacts)
        attempted_count += 1
        logger.info(
            "Attempting latest pairwise artifact load: persona=%s pair=%s judge=%s artifact=%s",
            config.persona,
            pair.label,
            judge,
            latest,
        )
        try:
            records = [
                _apply_canonical_user_id_to_record(
                    record=record,
                    persona_alias=str(config.persona),
                    alias_to_user_id=alias_to_user_id,
                )
                for record in load_pairwise_json(latest)
            ]
            indexed = index_records_by_task_id(records)
        except Exception as exc:
            wrapped = wrap_pairwise_artifact_load_error(
                exc,
                context=PairwiseArtifactLoadContext(
                    artifact_path=str(latest),
                    failure_stage="streamlit_cached_load_latest_records",
                    judge_token=str(judge),
                    persona=str(config.persona),
                    prompt_type=str(config.prompt_type),
                    pairwise_judgment_type=str(config.pairwise_judgment_type),
                    generator_model=str(config.generator_model),
                    filter_model=str(config.filter_model),
                ),
                message="Streamlit failed to load the latest discovered pairwise artifact.",
            )
            logger.warning("%s", wrapped)
            failure_messages.append(str(wrapped))
            continue
        index_by_judge[judge] = indexed
        artifact_by_judge[judge] = str(latest)
        loaded_count += 1
        logger.info(
            "Loaded latest pairwise artifact: persona=%s pair=%s judge=%s artifact=%s task_count=%d attempted=%d loaded=%d failed=%d",
            config.persona,
            pair.label,
            judge,
            latest,
            len(indexed),
            attempted_count,
            loaded_count,
            attempted_count - loaded_count,
        )

    if not index_by_judge:
        raise ValueError(
            "Judges were discovered, but no readable pairwise artifacts were found."
        )
    if failure_messages:
        raise ValueError(
            "Some discovered pairwise artifacts failed to load in Streamlit latest-record view. "
            f"attempted={attempted_count} loaded={loaded_count} failed={attempted_count - loaded_count}. "
            f"Failures: {failure_messages}"
        )

    task_ids = build_task_id_union(index_by_judge)
    if not task_ids:
        raise ValueError("No task_ids could be built from the loaded judge artifacts.")

    return task_ids, index_by_judge, artifact_by_judge


@st.cache_data(show_spinner=False)
def _cached_load_indexed_records_for_artifact(
    artifact_path_str: str,
    base_dir_for_persona_inference: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Load one Stage-5b artifact and index its records by task_id.

    Args:
        artifact_path_str: Absolute artifact path string.
        base_dir_for_persona_inference: Base results directory used to infer the
            persona alias for canonical ``user_id`` normalization.

    Returns:
        Dict[str, Dict[str, Any]]: task_id -> record mapping.
    """

    path = Path(str(artifact_path_str)).expanduser().resolve()
    try:
        records = load_pairwise_json(path)
    except Exception as exc:
        raise wrap_pairwise_artifact_load_error(
            exc,
            context=PairwiseArtifactLoadContext(
                artifact_path=str(path),
                failure_stage="streamlit_cached_load_indexed_records",
            ),
            message="Streamlit failed while loading the selected pairwise artifact.",
        ) from exc
    if base_dir_for_persona_inference:
        records = _canonicalize_artifact_records_for_weight_lookup(
            records=records,
            artifact_path_str=str(path),
            base_dir_str=base_dir_for_persona_inference,
        )
    return index_records_by_task_id(records)


@st.cache_data(show_spinner=False)
def _cached_load_dimension_weights_by_user(
    config_paths: Tuple[str, ...],
    selection_file_mtime: float,
) -> Dict[str, Dict[str, float]]:
    """
    Load persona-driven pairwise dimension weights (cached).

    Args:
        config_paths: Absolute YAML paths to persona configs.
        selection_file_mtime: mtime of the selection file used to invalidate cache.

    Returns:
        Dict[str, Dict[str, float]]: user_id -> {pairwise_dim -> weight}.
    """

    _ = float(selection_file_mtime or 0.0)
    paths = [str(Path(p).expanduser().resolve()) for p in (config_paths or ())]
    logger.info("loading persona dimension weights from config paths: %s", paths)
    return load_persona_pairwise_dimension_weights(paths, log=logger)


@st.cache_data(show_spinner=False)
def _cached_load_objective_lookup(
    base_dir_str: str,
) -> Optional[Dict]:
    """
    Scan for Stage 4 objective results and build an objective lookup dict.

    The lookup maps ``(sample_key, variant_label, model_name)`` to a dict
    containing ``pass_at_1`` and ``plus_pass_at_1`` values.

    Args:
        base_dir_str: Root results directory (e.g. ``Runs/new_gpt5_100_samples``).

    Returns:
        Optional[Dict]: The objective lookup dict, or None if no data found.
    """
    base_dir = Path(base_dir_str).expanduser().resolve()
    loader = AnalysisInputLoader(logger=logger)

    obj_frames = []
    obj_dir = base_dir / "4_objective_evaluation"
    if not obj_dir.exists():
        for persona_dir in sorted(base_dir.iterdir()):
            candidate = persona_dir / "4_objective_evaluation"
            if candidate.exists() and candidate.is_dir():
                for obj_file in sorted(candidate.rglob("*.jsonl")):
                    try:
                        df = loader.load_objective_results(str(obj_file))
                        if df is not None and not df.empty:
                            obj_frames.append(df)
                    except Exception:
                        pass
    else:
        for obj_file in sorted(obj_dir.rglob("*.jsonl")):
            try:
                df = loader.load_objective_results(str(obj_file))
                if df is not None and not df.empty:
                    obj_frames.append(df)
            except Exception:
                pass

    if not obj_frames:
        return None

    all_obj_df = pd.concat(obj_frames, ignore_index=True)
    if all_obj_df.empty:
        return None

    return _build_objective_lookup(all_obj_df)


@st.cache_data(show_spinner=False)
def _cached_load_stage6_pairwise_rows(
    artifact_path_str: str,
    tie_breaker_mode: str,
    omit_pairwise_keys: Tuple[str, ...],
    exclude_eval_parse_errors: bool,
    drop_invalid_samples: bool,
    judge_token: str,
    dimension_weighted_winner: bool,
    dimension_weight_config_paths: Tuple[str, ...],
    dimension_weight_selection_mtime: float,
    correctness_mode: str = "ignore",
    base_dir_for_objective: str = "",
    include_plus_correctness: bool = False,
    use_human_overall_choice: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load Stage-6-normalized pairwise rows from a Stage-5b artifact.

    This uses the same canonicalization logic as Stage 6 analysis
    (`AnalysisInputLoader.load_pairwise_results`).

    Args:
        artifact_path_str: Path to the Stage-5b pairwise JSON artifact.
        tie_breaker_mode: Stage-6 tie breaker mode ('strict' or 'finegrained').
        omit_pairwise_keys: Pairwise dimension keys to omit from overall-winner recomputation.
        exclude_eval_parse_errors: If True, exclude error dimensions from overall-winner vote.
        drop_invalid_samples: If True, drop rows that have no usable dimensions for overall.
        judge_token: Judge directory token this artifact was selected under.
        correctness_mode: Correctness handling mode (``"ignore"``, ``"dimension"``, ``"gate"``).
        base_dir_for_objective: Base directory for loading objective lookup (used as cache key).
        include_plus_correctness: If True, include plus-test correctness as a dimension.
        use_human_overall_choice: If True, human-judge rows use direct
            ``human_overall_choice`` and skip overall recomputation.

    Returns:
        List[Dict[str, Any]]: Row dicts from the Stage-6-normalized DataFrame.
    """

    loader = AnalysisInputLoader(logger=logger)
    try:
        df = loader.load_pairwise_results(
            artifact_path_str, tie_breaker_mode=tie_breaker_mode
        )
    except Exception as exc:
        raise wrap_pairwise_artifact_load_error(
            exc,
            context=PairwiseArtifactLoadContext(
                artifact_path=str(artifact_path_str),
                failure_stage="streamlit_cached_load_stage6_pairwise_rows",
                judge_token=str(judge_token),
                tie_breaker_mode=str(tie_breaker_mode),
            ),
            message="Streamlit failed while loading a Stage-6-normalized pairwise artifact.",
        ) from exc
    if df is None or df.empty:
        return []
    try:
        persona_alias = _infer_persona_alias_from_artifact_path(
            artifact_path_str, base_dir_for_objective
        )
        alias_to_user_id = _load_persona_weight_catalog()["alias_to_user_id"]
        if "source_persona" not in df.columns:
            df["source_persona"] = str(persona_alias)
        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].apply(
                lambda value: _resolve_weight_user_id(
                    persona_alias=persona_alias,
                    raw_user_id=value,
                    alias_to_user_id=alias_to_user_id,
                )
            )
        else:
            df["user_id"] = _resolve_weight_user_id(
                persona_alias=persona_alias,
                raw_user_id=None,
                alias_to_user_id=alias_to_user_id,
            )

        # Carry the judge token used by the UI/artifact selection.
        df["_judge_token"] = str(judge_token)

        # Fail-fast guard: a single artifact should correspond to a single judge model.
        judge_names = (
            df["judge_model_name"].dropna().astype(str).unique().tolist()
            if "judge_model_name" in df.columns
            else []
        )
        if len(judge_names) != 1:
            raise ValueError(
                "Stage-6-normalized artifact contains multiple distinct judge_model_name values. "
                f"artifact={artifact_path_str!r} judge_token={judge_token!r} judge_model_name_values={judge_names!r}"
            )

        obj_lookup = None
        if correctness_mode != "ignore" and base_dir_for_objective:
            obj_lookup = _cached_load_objective_lookup(base_dir_for_objective)

        if bool(dimension_weighted_winner):
            weights_by_user = _cached_load_dimension_weights_by_user(
                tuple(str(x) for x in (dimension_weight_config_paths or ())),
                float(dimension_weight_selection_mtime or 0.0),
            )
            df = recompute_pairwise_overall_winner_dimension_weighted(
                df,
                dimension_weights_by_user=weights_by_user,
                omit_pairwise_keys=set(omit_pairwise_keys),
                exclude_evaluation_error_dimensions=bool(exclude_eval_parse_errors),
                correctness_mode=correctness_mode,
                objective_lookup=obj_lookup,
                include_plus_correctness=include_plus_correctness,
            )
        else:
            df = recompute_pairwise_overall_winner(
                df,
                omit_pairwise_keys=set(omit_pairwise_keys),
                exclude_evaluation_error_dimensions=bool(exclude_eval_parse_errors),
                correctness_mode=correctness_mode,
                objective_lookup=obj_lookup,
                include_plus_correctness=include_plus_correctness,
            )
        if bool(use_human_overall_choice):
            df = _apply_direct_human_overall_choice_to_stage6_df(df)
        if bool(drop_invalid_samples):
            if "pairwise_valid_for_overall" not in df.columns:
                raise ValueError(
                    "Internal error: missing 'pairwise_valid_for_overall' after Stage-6-style recomputation. "
                    f"artifact={artifact_path_str!r}"
                )
            df = df[df["pairwise_valid_for_overall"]].copy()

        # Remove omitted per-dimension columns after winners were recomputed.
        df = apply_pairwise_dimension_omits(
            df, omit_pairwise_keys=set(omit_pairwise_keys)
        )

        # Stable per-sample identifier aligned with Stage 6 conventions.
        df["_stage6_sample_id"] = df.apply(
            lambda rr: stage6_sample_id_from_row(rr.to_dict()), axis=1
        )

        return df.to_dict(orient="records")
    except Exception as exc:
        raise wrap_pairwise_artifact_load_error(
            exc,
            context=PairwiseArtifactLoadContext(
                artifact_path=str(artifact_path_str),
                failure_stage="streamlit_stage6_pairwise_normalization",
                judge_token=str(judge_token),
                tie_breaker_mode=str(tie_breaker_mode),
            ),
            message="Streamlit failed while normalizing Stage-6 pairwise rows.",
        ) from exc


@st.cache_data(show_spinner=False)
def _cached_discover_judges_union(
    base_dir_str: str, cfg_dicts: List[Dict[str, Any]], pair_dicts: List[Dict[str, Any]]
) -> List[str]:
    """
    Discover the union of judge names across a set of (config, pair) selections.

    Args:
        base_dir_str: Base results directory as string.
        cfg_dicts: List of config dicts compatible with PairwiseConfigKey.
        pair_dicts: List of model pair dicts compatible with ModelPairKey.

    Returns:
        Sorted list of judge names.
    """

    base_dir = Path(base_dir_str).expanduser().resolve()
    judges: set[str] = set()
    for cfg in cfg_dicts:
        config = PairwiseConfigKey(**cfg)
        for p in pair_dicts:
            pair = ModelPairKey(**p)
            for judge in discover_judges_for_pair(base_dir, config, pair):
                judges.add(str(judge))
    return sorted(judges, key=str)


def _agg_item_id(meta: Dict[str, str]) -> str:
    """
    Build a stable, unique item identifier for aggregated tables.

    Args:
        meta: Metadata dict containing at least persona, generator_model, filter_model,
            prompt_type, model_1, model_2, task_id.

    Returns:
        Stable string key.
    """

    required = [
        "persona",
        "generator_model",
        "filter_model",
        "prompt_type",
        "pairwise_judgment_type",
        "model_1",
        "model_2",
        "task_id",
    ]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise ValueError(f"Missing required meta fields for agg item id: {missing}")
    return "|".join(str(meta[k]) for k in required)


def _agg_alignment_key(meta: Dict[str, str]) -> str:
    """Build an aggregate item key that ignores judgment type."""
    required = [
        "persona",
        "generator_model",
        "filter_model",
        "prompt_type",
        "model_1",
        "model_2",
        "task_id",
    ]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise ValueError(
            f"Missing required meta fields for agg alignment key: {missing}"
        )
    return "|".join(str(meta[k]) for k in required)


def _remap_item_mapping_keys(
    mapping: Dict[str, str],
    *,
    old_to_new_item_ids: Dict[str, str],
    field_name: str,
) -> Dict[str, str]:
    """Remap item ids in one judge->item mapping with conflict checks."""
    remapped: Dict[str, str] = {}
    for old_item_id, value in mapping.items():
        new_item_id = old_to_new_item_ids.get(str(old_item_id), str(old_item_id))
        existing = remapped.get(new_item_id)
        if existing is not None and existing != value:
            raise ValueError(
                "Conflicting values were produced while aligning aggregate "
                f"judgment types. field={field_name!r} item_id={new_item_id!r} "
                f"existing={existing!r} new={value!r}"
            )
        remapped[new_item_id] = value
    return remapped


def _align_aggregate_item_judgment_types_to_human_scope(
    item_meta_by_id: Dict[str, Dict[str, Any]],
    display_outcomes_by_judge: Dict[str, Dict[str, str]],
    positional_outcomes_by_judge: Dict[str, Dict[str, str]],
    human_confidence_by_judge: Dict[str, Dict[str, str]],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
]:
    """
    Align aggregate item judgment types to the human-derived scope.

    Human rows default to ``persona`` unless their existing item metadata is
    explicitly ``general_user``. Matching LLM rows on the same aggregate item
    are then moved into the same judgment-type bucket.
    """
    if not item_meta_by_id:
        return (
            item_meta_by_id,
            display_outcomes_by_judge,
            positional_outcomes_by_judge,
            human_confidence_by_judge,
        )

    rows: List[Dict[str, str]] = []
    for judge, mapping in display_outcomes_by_judge.items():
        for item_id in mapping.keys():
            meta = item_meta_by_id.get(str(item_id))
            if not isinstance(meta, dict):
                continue
            rows.append(
                {
                    "item_id": str(item_id),
                    "alignment_key": _agg_alignment_key(meta),
                    "judge": str(judge),
                    "pairwise_judgment_type": str(
                        meta.get("pairwise_judgment_type", "persona")
                    ),
                }
            )
    if not rows:
        return (
            item_meta_by_id,
            display_outcomes_by_judge,
            positional_outcomes_by_judge,
            human_confidence_by_judge,
        )

    aligned = align_pairwise_judgment_types_to_human_scope(
        pd.DataFrame(rows),
        item_key_column="alignment_key",
        judge_column="judge",
        judgment_type_column="pairwise_judgment_type",
    )
    item_to_type = (
        aligned[["item_id", "pairwise_judgment_type"]]
        .drop_duplicates(subset=["item_id"], keep="last")
        .set_index("item_id")["pairwise_judgment_type"]
        .astype(str)
        .to_dict()
    )

    old_to_new_item_ids: Dict[str, str] = {}
    new_item_meta_by_id: Dict[str, Dict[str, Any]] = {}
    for old_item_id, meta in item_meta_by_id.items():
        target_type = item_to_type.get(
            str(old_item_id),
            str(meta.get("pairwise_judgment_type", "persona")),
        )
        new_meta = dict(meta)
        new_meta["pairwise_judgment_type"] = str(target_type)
        new_item_id = _agg_item_id(
            {
                "persona": str(new_meta["persona"]),
                "generator_model": str(new_meta["generator_model"]),
                "filter_model": str(new_meta["filter_model"]),
                "prompt_type": str(new_meta["prompt_type"]),
                "pairwise_judgment_type": str(new_meta["pairwise_judgment_type"]),
                "model_1": str(new_meta["model_1"]),
                "model_2": str(new_meta["model_2"]),
                "task_id": str(new_meta["task_id"]),
            }
        )
        old_to_new_item_ids[str(old_item_id)] = new_item_id
        existing_meta = new_item_meta_by_id.get(new_item_id)
        if existing_meta is not None and existing_meta != new_meta:
            raise ValueError(
                "Conflicting aggregate item metadata was produced while aligning "
                f"judgment types. item_id={new_item_id!r}"
            )
        new_item_meta_by_id[new_item_id] = new_meta

    new_display = {
        str(judge): _remap_item_mapping_keys(
            mapping,
            old_to_new_item_ids=old_to_new_item_ids,
            field_name="display_outcomes_by_judge",
        )
        for judge, mapping in display_outcomes_by_judge.items()
    }
    new_positional = {
        str(judge): _remap_item_mapping_keys(
            mapping,
            old_to_new_item_ids=old_to_new_item_ids,
            field_name="positional_outcomes_by_judge",
        )
        for judge, mapping in positional_outcomes_by_judge.items()
    }
    new_human_confidence = {
        str(judge): _remap_item_mapping_keys(
            mapping,
            old_to_new_item_ids=old_to_new_item_ids,
            field_name="human_confidence_by_judge",
        )
        for judge, mapping in human_confidence_by_judge.items()
    }
    return (
        new_item_meta_by_id,
        new_display,
        new_positional,
        new_human_confidence,
    )


def _map_display_winner_to_positional(
    *,
    display_winner: str,
    model_1: str,
    model_2: str,
) -> str:
    """
    Map a display winner name to a position-based label for global agreement.

    Args:
        display_winner: Winner display value (model name or 'tie').
        model_1: First model in the pair (row-level).
        model_2: Second model in the pair (row-level).

    Returns:
        One of {'model_1','model_2','tie'}.
    """

    if display_winner == "tie":
        return "tie"
    if display_winner == model_1:
        return "model_1"
    if display_winner == model_2:
        return "model_2"
    return "tie"


def _flip_model_winner_label(label: object) -> str:
    """
    Flip a Stage-6 winner label when model A/B are swapped.

    Args:
        label: Winner label value from Stage-6-shaped frames.

    Returns:
        str: One of {'model_a','model_b','tie'}.
    """

    token = str(label or "tie").strip()
    if token == "model_a":
        return "model_b"
    if token == "model_b":
        return "model_a"
    return "tie"


def _build_aggregate_dimension_sample_frame_for_model_pair(
    *,
    selected_model_pair: Optional[str],
    item_ids_for_pair: List[str],
    item_meta_by_id: Dict[str, Dict[str, Any]],
    judges_effective: List[str],
    artifacts_used_rows: List[Dict[str, Any]],
    tie_breaker_mode: str,
    stage6_tie_breaker_mode: str,
    omit_pairwise_keys: Sequence[str],
    exclude_eval_parse_errors: bool,
    drop_invalid_samples: bool,
    dimension_weighted_overall_winner: bool,
    dimension_weight_config_paths: Sequence[str],
    dimension_weight_selection_mtime: float,
    correctness_mode: str = "ignore",
    base_dir_str: str = "",
    include_plus_correctness: bool = False,
    streamlit_objective_lookup: Optional[Dict] = None,
    use_human_overall_choice: bool = False,
) -> pd.DataFrame:
    """
    Build a Stage-6-shaped per-sample frame with per-dimension winner labels.

    This frame is designed to be fed into `compute_dimension_win_rates(...)`.

    Args:
        selected_model_pair: Optional model-pair label (e.g. "model1_vs_model2").
            When omitted, build rows across the current filtered aggregate scope.
        item_ids_for_pair: Aggregate item ids in scope for the current filters.
        item_meta_by_id: item_id -> meta dict for aggregate mode.
        judges_effective: Judge tokens included after judge filtering (e.g. present_only).
        artifacts_used_rows: Rows describing which artifacts were used for each (cfg,pair,judge).
        tie_breaker_mode: UI tie breaker selection. If it starts with "stage6_", use Stage-6-normalized rows.
        stage6_tie_breaker_mode: Stage-6 tie breaker mode for normalized loading ('strict'|'finegrained').
        omit_pairwise_keys: Dimension keys to omit.
        exclude_eval_parse_errors: If True, exclude error dimensions from denominators.
        drop_invalid_samples: If True, drop samples that are invalid for overall (no usable dims).

    Returns:
        pd.DataFrame: One row per (comparison, judge) with dim_* columns.

    Raises:
        ValueError: If model ordering cannot be aligned or dimensions are missing.
    """

    omit_set = {str(x) for x in (omit_pairwise_keys or [])}
    dims_included = [d for d in PAIRWISE_DIMENSIONS if str(d) not in omit_set]
    alias_to_user_id = _load_persona_weight_catalog()["alias_to_user_id"]
    selected_model_pair_str = str(selected_model_pair or "").strip()
    scope_label = (
        f"model_pair={selected_model_pair_str!r}"
        if selected_model_pair_str
        else "aggregate_scope"
    )

    # NOTE: In aggregate mode, the same `task_id` commonly appears under multiple
    # configurations (different persona/prompt_type/gen/filter). Therefore, `task_id`
    # is NOT globally unique for a `model_pair` across the aggregate scope. We must
    # join records to metadata using both the artifact's config fields and task_id.
    def _meta_matches_artifact_row(
        meta: Dict[str, Any], artifact_row: Dict[str, Any]
    ) -> bool:
        return (
            str(meta.get("persona")) == str(artifact_row.get("persona"))
            and str(meta.get("generator_model"))
            == str(artifact_row.get("generator_model"))
            and str(meta.get("filter_model")) == str(artifact_row.get("filter_model"))
            and str(meta.get("prompt_type")) == str(artifact_row.get("prompt_type"))
            and str(meta.get("model_pair")) == str(artifact_row.get("model_pair"))
        )

    effective_judges = {str(j) for j in (judges_effective or [])}
    artifacts_for_pair = [
        r
        for r in (artifacts_used_rows or [])
        if (
            (not selected_model_pair_str)
            or (str(r.get("model_pair")) == selected_model_pair_str)
        )
        and str(r.get("judge_token")) in effective_judges
        and str(r.get("load_status", "loaded")).strip().lower() == "loaded"
    ]

    if not artifacts_for_pair:
        return pd.DataFrame()

    samples: List[Dict[str, Any]] = []

    if _is_stage6_tie_breaker(tie_breaker_mode):
        omit_tuple = tuple(sorted(str(x) for x in omit_set))
        for r in artifacts_for_pair:
            artifact_path = str(r.get("artifact_path", "")).strip()
            judge_token = str(r.get("judge_token", "")).strip()
            if not artifact_path or not judge_token:
                continue

            # Restrict to items belonging to this exact (config, pair) artifact row.
            item_ids_scoped = [
                item_id
                for item_id in item_ids_for_pair
                if isinstance(item_meta_by_id.get(item_id), dict)
                and _meta_matches_artifact_row(item_meta_by_id[item_id], r)
            ]
            if not item_ids_scoped:
                continue
            task_id_to_meta: Dict[str, Tuple[Dict[str, Any], str]] = {}
            for item_id in item_ids_scoped:
                meta = item_meta_by_id[item_id]
                task_id = str(meta.get("task_id", "")).strip()
                if not task_id:
                    continue
                # Within a single config scope, task_id should be unique.
                if task_id in task_id_to_meta and task_id_to_meta[task_id][0] != meta:
                    raise ValueError(
                        "Conflicting aggregate metadata for same task_id within a single artifact scope. "
                        f"task_id={task_id!r} {scope_label} artifact={artifact_path!r}"
                    )
                task_id_to_meta[task_id] = (meta, str(item_id))
            keep_task_ids = set(task_id_to_meta.keys())
            if not keep_task_ids:
                continue

            rows = _cached_load_stage6_pairwise_rows(
                artifact_path,
                tie_breaker_mode=str(stage6_tie_breaker_mode),
                omit_pairwise_keys=omit_tuple,
                exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                drop_invalid_samples=bool(drop_invalid_samples),
                judge_token=str(judge_token),
                dimension_weighted_winner=bool(dimension_weighted_overall_winner),
                dimension_weight_config_paths=tuple(
                    str(x) for x in (dimension_weight_config_paths or [])
                ),
                dimension_weight_selection_mtime=float(
                    dimension_weight_selection_mtime or 0.0
                ),
                correctness_mode=correctness_mode,
                base_dir_for_objective=base_dir_str,
                include_plus_correctness=bool(include_plus_correctness),
                use_human_overall_choice=bool(use_human_overall_choice),
            )
            for row in rows:
                task_id = str(row.get("_stage6_sample_id", "")).strip()
                if task_id not in keep_task_ids:
                    continue
                meta, item_id = task_id_to_meta[task_id]
                model_1 = str(meta.get("model_1", "")).strip()
                model_2 = str(meta.get("model_2", "")).strip()
                if not model_1 or not model_2:
                    raise ValueError(
                        "Aggregate meta is missing model_1/model_2 for a kept task_id. "
                        f"task_id={task_id!r}"
                    )

                row_model_a = str(row.get("model_a_name", "")).strip()
                row_model_b = str(row.get("model_b_name", "")).strip()
                if not row_model_a or not row_model_b:
                    raise ValueError(
                        "Stage-6 row is missing model_a_name/model_b_name. "
                        f"artifact={artifact_path!r} judge={judge_token!r} task_id={task_id!r}"
                    )

                if row_model_a == model_1 and row_model_b == model_2:
                    flip = False
                elif row_model_a == model_2 and row_model_b == model_1:
                    flip = True
                else:
                    raise ValueError(
                        "Cannot align Stage-6 row model ordering to aggregate model_pair ordering. "
                        f"artifact={artifact_path!r} judge={judge_token!r} task_id={task_id!r} "
                        f"row_models=({row_model_a!r},{row_model_b!r}) meta_models=({model_1!r},{model_2!r})"
                    )

                rec: Dict[str, Any] = {
                    "item_id": str(item_id),
                    "user_id": _resolve_weight_user_id(
                        persona_alias=str(meta.get("persona", "")),
                        raw_user_id=row.get(
                            "user_id", meta.get("canonical_user_id", "")
                        ),
                        alias_to_user_id=alias_to_user_id,
                    ),
                    "variant_label": str(meta.get("prompt_type", "")),
                    "task_id": task_id,
                    "variant_id": str(meta.get("variant_id", "") or ""),
                    "model_a_name": model_1,
                    "model_b_name": model_2,
                    "model_pair": str(meta.get("model_pair", "") or selected_model_pair_str),
                    "judge_model_name": str(judge_token),
                }
                for dim in dims_included:
                    w_col = f"dim_{dim}_winner_label"
                    b_col = f"dim_{dim}_bias_detected"
                    c_col = f"dim_{dim}_confidence"
                    r_col = f"dim_{dim}_rationale"
                    if w_col not in row:
                        raise ValueError(
                            "Missing required per-dimension winner label in Stage-6 row. "
                            f"artifact={artifact_path!r} judge={judge_token!r} task_id={task_id!r} missing={w_col!r}"
                        )
                    winner_label = str(row.get(w_col, "tie"))
                    if flip:
                        winner_label = _flip_model_winner_label(winner_label)
                    rec[w_col] = winner_label
                    rec[b_col] = bool(row.get(b_col, False))
                    rec[c_col] = str(row.get(c_col, "low"))
                    # Keep rationale column when present; we may drop all rationale cols later
                    # to disable error-exclusion from denominators.
                    rec[r_col] = row.get(r_col)
                samples.append(rec)
    else:
        dim_mode = _dimension_tie_breaker_mode(tie_breaker_mode)
        weights_by_user: Optional[Dict[str, Dict[str, float]]] = None
        if bool(dimension_weighted_overall_winner):
            weights_by_user = _cached_load_dimension_weights_by_user(
                tuple(str(x) for x in (dimension_weight_config_paths or [])),
                float(dimension_weight_selection_mtime or 0.0),
            )
        for r in artifacts_for_pair:
            artifact_path = str(r.get("artifact_path", "")).strip()
            judge_token = str(r.get("judge_token", "")).strip()
            if not artifact_path or not judge_token:
                continue

            # Restrict to items belonging to this exact (config, pair) artifact row.
            item_ids_scoped = [
                item_id
                for item_id in item_ids_for_pair
                if isinstance(item_meta_by_id.get(item_id), dict)
                and _meta_matches_artifact_row(item_meta_by_id[item_id], r)
            ]
            if not item_ids_scoped:
                continue
            task_id_to_meta: Dict[str, Tuple[Dict[str, Any], str]] = {}
            for item_id in item_ids_scoped:
                meta = item_meta_by_id[item_id]
                task_id = str(meta.get("task_id", "")).strip()
                if not task_id:
                    continue
                if task_id in task_id_to_meta and task_id_to_meta[task_id][0] != meta:
                    raise ValueError(
                        "Conflicting aggregate metadata for same task_id within a single artifact scope. "
                        f"task_id={task_id!r} {scope_label} artifact={artifact_path!r}"
                    )
                task_id_to_meta[task_id] = (meta, str(item_id))
            keep_task_ids = set(task_id_to_meta.keys())
            if not keep_task_ids:
                continue

            indexed = _cached_load_indexed_records_for_artifact(
                artifact_path,
                base_dir_str,
            )
            # Iterate over the kept task ids to avoid scanning large artifacts repeatedly.
            for task_id in keep_task_ids:
                record = indexed.get(task_id)
                if not isinstance(record, dict):
                    continue
                record = _apply_canonical_user_id_to_record(
                    record=record,
                    persona_alias=str(task_id_to_meta[task_id][0].get("persona", "")),
                    alias_to_user_id=alias_to_user_id,
                )

                if drop_invalid_samples and not (
                    bool(use_human_overall_choice)
                    and _record_uses_human_judge(record, judge_token=judge_token)
                ):
                    corr_a, corr_b, plus_a, plus_b = _get_record_correctness(
                        record, correctness_mode, streamlit_objective_lookup
                    )
                    if bool(dimension_weighted_overall_winner):
                        if weights_by_user is None:
                            raise ValueError(
                                "Internal error: weights_by_user was not loaded for "
                                "dimension-weighted overall recomputation."
                            )
                        (
                            _overall,
                            _score_a,
                            _score_b,
                            _score_tie,
                            valid_for_overall,
                        ) = recompute_overall_from_dimensions_with_controls_weighted(
                            record=record,
                            mode=dim_mode,
                            omit_pairwise_keys=omit_set,
                            exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                            dimension_weights_by_user=weights_by_user,
                            correctness_mode=correctness_mode,
                            correctness_a=corr_a,
                            correctness_b=corr_b,
                            plus_correctness_a=plus_a,
                            plus_correctness_b=plus_b,
                            include_plus_correctness=bool(include_plus_correctness),
                        )
                    else:
                        _overall, _a, _b, _t, valid_for_overall = (
                            recompute_overall_from_dimensions_with_controls(
                                record=record,
                                mode=dim_mode,
                                omit_pairwise_keys=omit_set,
                                exclude_eval_parse_errors=bool(
                                    exclude_eval_parse_errors
                                ),
                                correctness_mode=correctness_mode,
                                correctness_a=corr_a,
                                correctness_b=corr_b,
                                plus_correctness_a=plus_a,
                                plus_correctness_b=plus_b,
                                include_plus_correctness=bool(include_plus_correctness),
                            )
                        )
                    if not valid_for_overall:
                        continue

                meta, item_id = task_id_to_meta[task_id]
                model_1 = str(meta.get("model_1", "")).strip()
                model_2 = str(meta.get("model_2", "")).strip()
                if not model_1 or not model_2:
                    raise ValueError(
                        "Aggregate meta is missing model_1/model_2 for a kept task_id. "
                        f"task_id={task_id!r}"
                    )

                rec_model_a = str(record.get("model_a_name", model_1))
                rec_model_b = str(record.get("model_b_name", model_2))

                dim_results = record.get("dimension_results") or {}
                if not isinstance(dim_results, dict):
                    continue

                rec: Dict[str, Any] = {
                    "item_id": str(item_id),
                    "user_id": _resolve_weight_user_id(
                        persona_alias=str(meta.get("persona", "")),
                        raw_user_id=record.get(
                            "user_id", meta.get("canonical_user_id", "")
                        ),
                        alias_to_user_id=alias_to_user_id,
                    ),
                    "variant_label": str(meta.get("prompt_type", "")),
                    "task_id": str(task_id),
                    "variant_id": str(meta.get("variant_id", "") or ""),
                    "model_a_name": model_1,
                    "model_b_name": model_2,
                    "model_pair": str(meta.get("model_pair", "") or selected_model_pair_str),
                    "judge_model_name": str(judge_token),
                }

                for dim in dims_included:
                    dim_payload = dim_results.get(dim)
                    if not isinstance(dim_payload, dict):
                        raise ValueError(
                            "Missing required dimension payload in Stage-5b record. "
                            f"artifact={artifact_path!r} judge={judge_token!r} task_id={task_id!r} missing_dim={dim!r}"
                        )

                    winner_name, confidence = _recompute_dimension_winner(
                        dim_result=dim_payload,
                        mode=dim_mode,
                        model_a_name=rec_model_a,
                        model_b_name=rec_model_b,
                    )
                    if winner_name == "tie":
                        winner_label = "tie"
                    elif winner_name == model_1:
                        winner_label = "model_a"
                    elif winner_name == model_2:
                        winner_label = "model_b"
                    else:
                        raise ValueError(
                            "Dimension winner name cannot be mapped to aggregate model_1/model_2. "
                            f"artifact={artifact_path!r} judge={judge_token!r} task_id={task_id!r} "
                            f"winner_name={winner_name!r} model_1={model_1!r} model_2={model_2!r}"
                        )

                    rec[f"dim_{dim}_winner_label"] = winner_label
                    rec[f"dim_{dim}_bias_detected"] = bool(
                        dim_payload.get("position_bias_detected", False)
                    )
                    rec[f"dim_{dim}_confidence"] = str(confidence or "low")
                    rec[f"dim_{dim}_rationale"] = dim_payload.get("rationale")
                samples.append(rec)

    df = pd.DataFrame(samples)
    if df.empty:
        return df

    if not exclude_eval_parse_errors:
        rationale_cols = [
            c for c in df.columns if c.startswith("dim_") and c.endswith("_rationale")
        ]
        if rationale_cols:
            df = df.drop(columns=rationale_cols)

    return df


@st.cache_data(show_spinner=False)
def _cached_load_aggregated_outcomes(
    base_dir_str: str,
    cfg_dicts: List[Dict[str, Any]],
    pair_dicts: List[Dict[str, Any]],
    judges: List[str],
    tie_breaker_mode: str,
    selection_path_str: str,
    selection_file_mtime: float,
    stage6_tie_breaker_mode: str,
    stage6_omit_pairwise_keys: Tuple[str, ...],
    stage6_exclude_eval_parse_errors: bool,
    stage6_drop_invalid_samples: bool,
    dimension_weighted_overall_winner: bool,
    dimension_weight_config_paths: Tuple[str, ...],
    dimension_weight_selection_mtime: float,
    correctness_mode: str = "ignore",
    include_plus_correctness: bool = False,
    use_human_overall_choice: bool = False,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    List[Dict[str, Any]],
]:
    """
    Load aggregated overall-winner outcomes across many (config, pair) selections.

    This loader is intentionally narrow: it computes only the overall winner outcomes
    needed for the All-samples tab and drops large text fields to keep memory low.

    Args:
        base_dir_str: Base results directory as string.
        cfg_dicts: List of config dicts compatible with PairwiseConfigKey.
        pair_dicts: List of model pair dicts compatible with ModelPairKey.
        judges: Judge names to attempt to load (missing artifacts are skipped).
        tie_breaker_mode: UI tie-breaker token. If it starts with `stage6_`, use the
            Stage-6-normalized loader; otherwise use the Streamlit swap-aware recompute.
        selection_path_str: Path to the artifact selection JSON file (per run directory).
        selection_file_mtime: Selection file modification time (used to invalidate cache).
        stage6_tie_breaker_mode: Stage-6 tie breaker mode ('strict' or 'finegrained').
        stage6_omit_pairwise_keys: Pairwise dimension keys to omit.
        stage6_exclude_eval_parse_errors: If True, exclude error dimensions from overall.
        stage6_drop_invalid_samples: If True, drop samples with no usable dimensions.

    Returns:
        Tuple of:
        - item_meta_by_id: item_id -> meta dict (persona/gen/filter/prompt/model_1/model_2/model_pair/task_id)
        - display_outcomes_by_judge: judge -> (item_id -> winner model name or 'tie')
        - positional_outcomes_by_judge: judge -> (item_id -> 'model_1'/'model_2'/'tie')
        - human_confidence_by_judge: human judge -> (item_id -> human_overall_confidence)
        - artifacts_used_rows: list of dict rows describing which artifact was used per (cfg,pair,judge)
    """

    base_dir = Path(base_dir_str).expanduser().resolve()
    configs = [PairwiseConfigKey(**d) for d in cfg_dicts]
    pairs = [ModelPairKey(**d) for d in pair_dicts]
    judge_list = [str(j) for j in judges]
    persona_weight_catalog = _load_persona_weight_catalog()
    alias_to_user_id = persona_weight_catalog["alias_to_user_id"]

    item_meta_by_id: Dict[str, Dict[str, Any]] = {}
    display_outcomes_by_judge: Dict[str, Dict[str, str]] = {j: {} for j in judge_list}
    positional_outcomes_by_judge: Dict[str, Dict[str, str]] = {
        j: {} for j in judge_list
    }
    human_confidence_by_judge: Dict[str, Dict[str, str]] = {j: {} for j in judge_list}
    artifacts_used_rows: List[Dict[str, Any]] = []

    # Load selections inside the cached function; include mtime in the cache key to refresh.
    _ = float(selection_file_mtime or 0.0)
    selection_path = Path(selection_path_str).expanduser().resolve()
    selections_by_key: Dict[str, dict] = {}
    if selection_path.exists():
        selections_by_key = load_pairwise_artifact_selections(selection_path)

    weights_by_user: Optional[Dict[str, Dict[str, float]]] = None
    if bool(dimension_weighted_overall_winner):
        weights_by_user = _cached_load_dimension_weights_by_user(
            tuple(str(x) for x in (dimension_weight_config_paths or ())),
            float(dimension_weight_selection_mtime or 0.0),
        )

    any_loaded = False
    attempted_count = 0
    loaded_count = 0
    failed_count = 0
    for config in configs:
        prompt_label = format_prompt_type(config.prompt_type)
        for pair in pairs:
            model_1 = str(pair.model_a)
            model_2 = str(pair.model_b)
            model_pair = f"{model_1}_vs_{model_2}"

            for judge in judge_list:
                artifacts = list_pairwise_artifacts_for_pair_and_judge(
                    base_dir, config, pair, judge
                )
                if not artifacts:
                    logger.debug(
                        "Aggregate loader found no artifacts: persona=%s pair=%s judge=%s prompt_type=%s judgment_type=%s",
                        config.persona,
                        model_pair,
                        judge,
                        prompt_label,
                        config.pairwise_judgment_type,
                    )
                    continue
                stage6_default = choose_stage6_default_artifact(artifacts)
                stage6_default_str = str(stage6_default)

                key = PairwiseArtifactSelectionKey(
                    persona=str(config.persona),
                    generator_model=str(config.generator_model),
                    filter_model=str(config.filter_model),
                    prompt_type=str(prompt_label),
                    pairwise_judgment_type=str(config.pairwise_judgment_type),
                    model_a=str(pair.model_a),
                    model_b=str(pair.model_b),
                    judge_model=str(judge),
                )
                saved_selected = get_saved_selected_path(selections_by_key, key)
                selected_path_str = stage6_default_str
                selected_reason = "stage6_default"
                if saved_selected is not None:
                    saved_resolved = str(Path(saved_selected).expanduser().resolve())
                    if not Path(saved_resolved).exists():
                        raise FileNotFoundError(
                            "Saved artifact selection path does not exist. "
                            f"key={key.to_string()!r} saved={saved_resolved!r}"
                        )
                    candidates = {str(p) for p in artifacts}
                    if saved_resolved not in candidates:
                        raise ValueError(
                            "Saved artifact selection does not match current candidates. "
                            f"key={key.to_string()!r} saved={saved_resolved!r}"
                        )
                    selected_path_str = saved_resolved
                    selected_reason = "saved"

                attempted_count += 1
                logger.info(
                    "Attempting aggregate pairwise artifact load: persona=%s pair=%s judge=%s artifact=%s reason=%s tie_breaker_mode=%s",
                    config.persona,
                    model_pair,
                    judge,
                    selected_path_str,
                    selected_reason,
                    tie_breaker_mode,
                )
                try:
                    if _is_stage6_tie_breaker(tie_breaker_mode):
                        rows = _cached_load_stage6_pairwise_rows(
                            str(selected_path_str),
                            tie_breaker_mode=str(stage6_tie_breaker_mode),
                            omit_pairwise_keys=tuple(stage6_omit_pairwise_keys),
                            exclude_eval_parse_errors=bool(
                                stage6_exclude_eval_parse_errors
                            ),
                            drop_invalid_samples=bool(stage6_drop_invalid_samples),
                            judge_token=str(judge),
                            dimension_weighted_winner=bool(
                                dimension_weighted_overall_winner
                            ),
                            dimension_weight_config_paths=tuple(
                                str(x) for x in (dimension_weight_config_paths or ())
                            ),
                            dimension_weight_selection_mtime=float(
                                dimension_weight_selection_mtime or 0.0
                            ),
                            correctness_mode=correctness_mode,
                            base_dir_for_objective=base_dir_str,
                            include_plus_correctness=bool(include_plus_correctness),
                            use_human_overall_choice=bool(use_human_overall_choice),
                        )
                        for row in rows:
                            sample_id = str(
                                row.get("_stage6_sample_id")
                                or stage6_sample_id_from_row(row)
                            )
                            meta: Dict[str, Any] = {
                                "persona": str(config.persona),
                                "canonical_user_id": _resolve_weight_user_id(
                                    persona_alias=str(config.persona),
                                    raw_user_id=row.get("user_id"),
                                    alias_to_user_id=alias_to_user_id,
                                ),
                                "generator_model": str(config.generator_model),
                                "filter_model": str(config.filter_model),
                                "prompt_type": str(prompt_label),
                                "pairwise_judgment_type": str(
                                    config.pairwise_judgment_type
                                ),
                                "model_1": model_1,
                                "model_2": model_2,
                                "model_pair": model_pair,
                                "task_id": str(sample_id),
                                "raw_task_id": str(row.get("raw_task_id", "") or ""),
                                "base_task_id": str(row.get("task_id", "") or ""),
                                "variant_id": str(row.get("variant_id", "") or ""),
                                "variant_label": str(row.get("variant_label", "") or ""),
                            }
                            item_id = _agg_item_id(
                                {
                                    "persona": str(meta["persona"]),
                                    "generator_model": str(meta["generator_model"]),
                                    "filter_model": str(meta["filter_model"]),
                                    "prompt_type": str(meta["prompt_type"]),
                                    "pairwise_judgment_type": str(
                                        meta["pairwise_judgment_type"]
                                    ),
                                    "model_1": str(meta["model_1"]),
                                    "model_2": str(meta["model_2"]),
                                    "task_id": str(meta["task_id"]),
                                }
                            )
                            item_meta_by_id[item_id] = meta

                            winner = stage6_display_winner_name_from_row(row)
                            display_outcomes_by_judge[judge][item_id] = winner
                            positional_outcomes_by_judge[judge][item_id] = (
                                _map_display_winner_to_positional(
                                    display_winner=winner,
                                    model_1=model_1,
                                    model_2=model_2,
                                )
                            )
                            if is_human_judge_token(judge):
                                confidence = str(
                                    row.get("human_overall_confidence", "") or ""
                                ).strip()
                                if confidence:
                                    human_confidence_by_judge[judge][item_id] = (
                                        confidence
                                    )
                    else:
                        records = load_pairwise_json(Path(str(selected_path_str)))
                        indexed = index_records_by_task_id(records)
                        for task_id, record in indexed.items():
                            record = _apply_canonical_user_id_to_record(
                                record=record,
                                persona_alias=str(config.persona),
                                alias_to_user_id=alias_to_user_id,
                            )
                            meta = {
                                "persona": str(config.persona),
                                "canonical_user_id": _resolve_weight_user_id(
                                    persona_alias=str(config.persona),
                                    raw_user_id=record.get("user_id"),
                                    alias_to_user_id=alias_to_user_id,
                                ),
                                "generator_model": str(config.generator_model),
                                "filter_model": str(config.filter_model),
                                "prompt_type": str(prompt_label),
                                "pairwise_judgment_type": str(
                                    config.pairwise_judgment_type
                                ),
                                "model_1": model_1,
                                "model_2": model_2,
                                "model_pair": model_pair,
                                "task_id": str(task_id),
                            }
                            item_id = _agg_item_id(
                                {
                                    "persona": str(meta["persona"]),
                                    "generator_model": str(meta["generator_model"]),
                                    "filter_model": str(meta["filter_model"]),
                                    "prompt_type": str(meta["prompt_type"]),
                                    "pairwise_judgment_type": str(
                                        meta["pairwise_judgment_type"]
                                    ),
                                    "model_1": str(meta["model_1"]),
                                    "model_2": str(meta["model_2"]),
                                    "task_id": str(meta["task_id"]),
                                }
                            )
                            item_meta_by_id[item_id] = meta
                            if is_human_judge_token(judge):
                                confidence = str(
                                    _record_public_metadata(record).get(
                                        "human_overall_confidence", ""
                                    )
                                    or ""
                                ).strip()
                                if confidence:
                                    human_confidence_by_judge[judge][item_id] = (
                                        confidence
                                    )

                            record_model_a = str(record.get("model_a_name", "model_a"))
                            record_model_b = str(record.get("model_b_name", "model_b"))
                            direct_human_winner = None
                            if bool(use_human_overall_choice):
                                direct_human_winner = (
                                    _resolve_direct_human_overall_winner(
                                        record, judge_token=str(judge)
                                    )
                                )
                            if direct_human_winner is not None:
                                winner = str(direct_human_winner)
                                display_outcomes_by_judge[judge][item_id] = winner
                                positional_outcomes_by_judge[judge][item_id] = (
                                    _map_display_winner_to_positional(
                                        display_winner=winner,
                                        model_1=model_1,
                                        model_2=model_2,
                                    )
                                )
                                continue
                            _obj_lookup = (
                                _cached_load_objective_lookup(base_dir_str)
                                if correctness_mode != "ignore"
                                else None
                            )
                            corr_a, corr_b, plus_a, plus_b = _get_record_correctness(
                                record, correctness_mode, _obj_lookup
                            )
                            if bool(dimension_weighted_overall_winner):
                                if weights_by_user is None:
                                    raise ValueError(
                                        "Internal error: weights_by_user was not loaded for "
                                        "dimension-weighted overall winner recomputation."
                                    )
                                (
                                    recomputed_overall,
                                    _score_a,
                                    _score_b,
                                    _score_tie,
                                    valid_for_overall,
                                ) = recompute_overall_from_dimensions_with_controls_weighted(
                                    record=record,
                                    mode=_dimension_tie_breaker_mode(tie_breaker_mode),
                                    omit_pairwise_keys=tuple(
                                        stage6_omit_pairwise_keys
                                    ),
                                    exclude_eval_parse_errors=bool(
                                        stage6_exclude_eval_parse_errors
                                    ),
                                    dimension_weights_by_user=weights_by_user,
                                    correctness_mode=correctness_mode,
                                    correctness_a=corr_a,
                                    correctness_b=corr_b,
                                    plus_correctness_a=plus_a,
                                    plus_correctness_b=plus_b,
                                    include_plus_correctness=bool(
                                        include_plus_correctness
                                    ),
                                )
                            else:
                                (
                                    recomputed_overall,
                                    _a,
                                    _b,
                                    _t,
                                    valid_for_overall,
                                ) = recompute_overall_from_dimensions_with_controls(
                                    record=record,
                                    mode=_dimension_tie_breaker_mode(tie_breaker_mode),
                                    omit_pairwise_keys=tuple(
                                        stage6_omit_pairwise_keys
                                    ),
                                    exclude_eval_parse_errors=bool(
                                        stage6_exclude_eval_parse_errors
                                    ),
                                    correctness_mode=correctness_mode,
                                    correctness_a=corr_a,
                                    correctness_b=corr_b,
                                    plus_correctness_a=plus_a,
                                    plus_correctness_b=plus_b,
                                    include_plus_correctness=bool(
                                        include_plus_correctness
                                    ),
                                )
                            if bool(stage6_drop_invalid_samples) and not bool(
                                valid_for_overall
                            ):
                                continue

                            winner = "tie"
                            if recomputed_overall == "tie":
                                winner = "tie"
                            elif recomputed_overall == record_model_a:
                                winner = record_model_a
                            elif recomputed_overall == record_model_b:
                                winner = record_model_b
                            elif str(recomputed_overall) in {model_1, model_2}:
                                winner = str(recomputed_overall)

                            if winner == record_model_a and record_model_a in {
                                model_1,
                                model_2,
                            }:
                                winner = record_model_a
                            if winner == record_model_b and record_model_b in {
                                model_1,
                                model_2,
                            }:
                                winner = record_model_b

                            display_outcomes_by_judge[judge][item_id] = winner
                            positional_outcomes_by_judge[judge][item_id] = (
                                _map_display_winner_to_positional(
                                    display_winner=winner,
                                    model_1=model_1,
                                    model_2=model_2,
                                )
                            )

                    any_loaded = True
                    loaded_count += 1
                    artifacts_used_rows.append(
                        _artifact_status_row(
                            persona=str(config.persona),
                            generator_model=str(config.generator_model),
                            filter_model=str(config.filter_model),
                            prompt_type=str(prompt_label),
                            pairwise_judgment_type=str(config.pairwise_judgment_type),
                            model_pair=model_pair,
                            judge_token=str(judge),
                            artifact_path=str(selected_path_str),
                            reason=str(selected_reason),
                            stage6_default_path=str(stage6_default_str),
                            load_status="loaded",
                        )
                    )
                    logger.info(
                        "Loaded aggregate pairwise artifact: persona=%s pair=%s judge=%s artifact=%s attempted=%d loaded=%d failed=%d",
                        config.persona,
                        model_pair,
                        judge,
                        selected_path_str,
                        attempted_count,
                        loaded_count,
                        failed_count,
                    )
                except Exception as exc:
                    failed_count += 1
                    wrapped = wrap_pairwise_artifact_load_error(
                        exc,
                        context=PairwiseArtifactLoadContext(
                            artifact_path=str(selected_path_str),
                            failure_stage="streamlit_cached_load_aggregated_outcomes",
                            judge_token=str(judge),
                            persona=str(config.persona),
                            prompt_type=str(prompt_label),
                            pairwise_judgment_type=str(config.pairwise_judgment_type),
                            generator_model=str(config.generator_model),
                            filter_model=str(config.filter_model),
                            tie_breaker_mode=str(tie_breaker_mode),
                        ),
                        message="Streamlit aggregate mode skipped a pairwise artifact because it failed to load.",
                    )
                    logger.warning("%s", wrapped)
                    artifacts_used_rows.append(
                        _artifact_status_row(
                            persona=str(config.persona),
                            generator_model=str(config.generator_model),
                            filter_model=str(config.filter_model),
                            prompt_type=str(prompt_label),
                            pairwise_judgment_type=str(config.pairwise_judgment_type),
                            model_pair=model_pair,
                            judge_token=str(judge),
                            artifact_path=str(selected_path_str),
                            reason=str(selected_reason),
                            stage6_default_path=str(stage6_default_str),
                            load_status="failed",
                            load_error=str(wrapped),
                        )
                    )
                    continue

    if not any_loaded:
        raise ValueError(
            "No readable pairwise artifacts found for the selected aggregate scope."
        )
    logger.info(
        "Aggregate pairwise load summary: attempted=%d loaded=%d failed=%d",
        attempted_count,
        loaded_count,
        failed_count,
    )

    (
        item_meta_by_id,
        display_outcomes_by_judge,
        positional_outcomes_by_judge,
        human_confidence_by_judge,
    ) = _align_aggregate_item_judgment_types_to_human_scope(
        item_meta_by_id,
        display_outcomes_by_judge,
        positional_outcomes_by_judge,
        human_confidence_by_judge,
    )

    return (
        item_meta_by_id,
        display_outcomes_by_judge,
        positional_outcomes_by_judge,
        human_confidence_by_judge,
        artifacts_used_rows,
    )


def _safe_get_record_field(
    record: Dict[str, Any], field: str, default: str = ""
) -> str:
    """
    Read a record field as string.

    Args:
        record: Pairwise record.
        field: Field name.
        default: Default value when missing.

    Returns:
        str: Field value rendered as string.
    """

    value = record.get(field, default)
    if value is None:
        return default
    return str(value)


def _render_text_block(label: str, text: str, height: int, key: str) -> None:
    """
    Render a scrollable text block.

    Args:
        label: UI label.
        text: Text content.
        height: Component height.
        key: Unique Streamlit widget key.
    """

    st.text_area(label, value=text or "", height=height, disabled=True, key=key)


def _render_agreement_mean_std(
    agreement_df: pd.DataFrame,
    *,
    title: str,
) -> None:
    """
    Render mean ± std summary for agreement metrics.

    Args:
        agreement_df: Output of `compute_judge_pair_agreement(...)`.
        title: Short label to distinguish the summary block.
    """

    if agreement_df is None or agreement_df.empty:
        st.caption(f"{title}: no judge-pair rows available.")
        return

    def _mean_std(series: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
        n = int(len(series))
        if n <= 0:
            return None, None, 0
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if n > 1 else 0.0
        return mean, std, n

    rows: List[Dict[str, Any]] = []
    metric_specs = [
        ("percent_agreement", "percent_agreement", "n_common"),
        (
            "percent_agreement_excl_ties",
            "percent_agreement_excl_ties",
            "n_common_excl_ties",
        ),
        ("cohens_kappa", "cohens_kappa", "n_common"),
        (
            "cohens_kappa_excl_ties",
            "cohens_kappa_excl_ties",
            "n_common_excl_ties",
        ),
    ]
    for source_col, label, item_count_col in metric_specs:
        series = pd.to_numeric(agreement_df.get(source_col), errors="coerce").dropna()
        mean_value, std_value, n_value = _mean_std(series)
        total_items_used = int(
            pd.to_numeric(agreement_df.get(item_count_col), errors="coerce").dropna().sum()
        )
        rows.append(
            {
                "metric": label,
                "mean": mean_value,
                "std": std_value,
                "n_pairs_used": n_value,
                "total_items_used": total_items_used,
            }
        )
    st.markdown(f"**Mean ± std across judge pairs** ({title})")
    st.dataframe(
        pd.DataFrame(rows),
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _render_per_human_llm_agreement_summary(
    agreement_df: pd.DataFrame,
    *,
    human_judges: Sequence[str],
    llm_judges: Sequence[str],
    title: str,
) -> None:
    """
    Render one summary row per human judge against all LLM judges.

    Args:
        agreement_df: Output of `compute_judge_pair_agreement(...)`.
        human_judges: Judge tokens recognized as human annotators.
        llm_judges: Judge tokens recognized as LLM judges.
        title: Short label to distinguish the summary block.
    """
    summary_df = summarize_human_vs_llm_agreement_means(
        agreement_df,
        human_judges=human_judges,
        llm_judges=llm_judges,
    )
    if summary_df.empty:
        st.caption(f"{title}: no human-vs-LLM judge-pair rows available.")
        return

    st.markdown(
        f"**Mean ± std across judge pairs for each human annotator** ({title})"
    )
    st.dataframe(
        summary_df,
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _sort_dimension_table(
    df: pd.DataFrame,
    *,
    omit_pairwise_keys: Sequence[str],
) -> pd.DataFrame:
    """
    Sort a dimension-keyed table using canonical pairwise dimension order.

    Args:
        df: Table with a `dimension` column.
        omit_pairwise_keys: Dimensions omitted from the current scope.

    Returns:
        pd.DataFrame: Sorted copy when a dimension column is present.
    """
    if df is None or df.empty or "dimension" not in df.columns:
        return df

    omit_set = {str(x) for x in (omit_pairwise_keys or [])}
    canonical_order = [
        str(dim) for dim in PAIRWISE_DIMENSIONS if str(dim) not in omit_set
    ]
    out = df.copy()
    out["dimension"] = pd.Categorical(
        out["dimension"].astype(str),
        categories=canonical_order,
        ordered=True,
    )
    return out.sort_values(by=["dimension"], ascending=True)


def _render_dimension_agreement_summary(
    agreement_df: pd.DataFrame,
    *,
    title: str,
    omit_pairwise_keys: Sequence[str],
) -> None:
    """
    Render one mean/std agreement summary row per dimension.

    Args:
        agreement_df: Output of `compute_dimension_judge_pair_agreement(...)`.
        title: Short label to distinguish the summary block.
        omit_pairwise_keys: Dimensions omitted from the current scope.
    """
    summary_df = summarize_dimension_agreement_mean_std(agreement_df)
    if summary_df.empty:
        st.caption(f"{title}: no per-dimension judge-pair rows available.")
        return

    st.markdown(f"**Mean ± std across judge pairs by dimension** ({title})")
    st.dataframe(
        _sort_dimension_table(summary_df, omit_pairwise_keys=omit_pairwise_keys),
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _render_dimension_per_human_llm_agreement_summary(
    agreement_df: pd.DataFrame,
    *,
    human_judges: Sequence[str],
    llm_judges: Sequence[str],
    title: str,
    omit_pairwise_keys: Sequence[str],
) -> None:
    """
    Render one summary row per (dimension, human judge) against all LLM judges.

    Args:
        agreement_df: Output of `compute_dimension_judge_pair_agreement(...)`.
        human_judges: Judge tokens recognized as human annotators.
        llm_judges: Judge tokens recognized as LLM judges.
        title: Short label to distinguish the summary block.
        omit_pairwise_keys: Dimensions omitted from the current scope.
    """
    summary_df = summarize_dimension_human_vs_llm_means(
        agreement_df,
        human_judges=human_judges,
        llm_judges=llm_judges,
    )
    if summary_df.empty:
        st.caption(f"{title}: no per-dimension human-vs-LLM judge-pair rows available.")
        return

    st.markdown(
        f"**Mean ± std across judge pairs for each human annotator by dimension** ({title})"
    )
    st.dataframe(
        _sort_dimension_table(summary_df, omit_pairwise_keys=omit_pairwise_keys),
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _build_pairwise_sample_df_from_named_outcomes(
    *,
    item_ids: Sequence[str],
    outcomes_by_judge: Dict[str, Dict[str, str]],
    user_id: str,
    variant_label: str,
    model_a_name: str,
    model_b_name: str,
    judge_column: str = "judge",
) -> pd.DataFrame:
    """
    Build a Stage-6-compatible pairwise sample DataFrame from model-name outcomes.

    This produces the minimal schema required by Stage 6 joint preference utilities:
    - user_id, variant_label, model_a_name, model_b_name, overall_winner_label

    Args:
        item_ids: Item/sample ids defining the analysis set.
        outcomes_by_judge: judge -> (item_id -> outcome) where outcome is
            {model_a_name, model_b_name, 'tie'}.
        user_id: Persona/user id label to attach (single-mode constant).
        variant_label: Prompt type label to attach (single-mode constant).
        model_a_name: Canonical model A name.
        model_b_name: Canonical model B name.
        judge_column: Judge column name to include for judge-split tables.

    Returns:
        pd.DataFrame: One row per (item_id, judge) where an outcome exists.

    Raises:
        ValueError: If an unexpected outcome value is observed.
    """

    uid = str(user_id)
    vlabel = str(variant_label)
    ma = str(model_a_name)
    mb = str(model_b_name)

    rows: List[Dict[str, Any]] = []
    for judge, mapping in outcomes_by_judge.items():
        if not isinstance(mapping, dict):
            continue
        j = str(judge)
        for item_id in item_ids:
            key = str(item_id)
            if key not in mapping:
                continue
            outcome = str(mapping.get(key))
            if outcome == "tie":
                overall = "tie"
            elif outcome == ma:
                overall = "model_a"
            elif outcome == mb:
                overall = "model_b"
            else:
                raise ValueError(
                    "Unexpected outcome value for named-outcome frame. "
                    f"item_id={key!r} judge={j!r} outcome={outcome!r} expected_one_of={[ma, mb, 'tie']!r}"
                )
            rows.append(
                {
                    "user_id": uid,
                    "variant_label": vlabel,
                    "task_id": str(key),
                    "variant_id": "",
                    "model_a_name": ma,
                    "model_b_name": mb,
                    "overall_winner_label": overall,
                    judge_column: j,
                }
            )

    return pd.DataFrame(rows)


def _build_pairwise_sample_df_from_positional_outcomes(
    *,
    item_ids: Sequence[str],
    outcomes_by_judge: Dict[str, Dict[str, str]],
    item_meta_by_id: Dict[str, Dict[str, Any]],
    judge_column: str = "judge",
) -> pd.DataFrame:
    """
    Build a Stage-6-compatible pairwise sample DataFrame from positional outcomes.

    This is used in aggregate mode where outcomes are position labels:
    - 'model_1' / 'model_2' / 'tie'

    We attach per-item metadata (persona/prompt/model_1/model_2) as the Stage-6
    required columns (user_id/variant_label/model_a_name/model_b_name).

    Args:
        item_ids: Item ids defining the analysis set.
        outcomes_by_judge: judge -> (item_id -> positional outcome).
        item_meta_by_id: item_id -> metadata dict containing at least:
            - persona
            - prompt_type
            - model_1
            - model_2
        judge_column: Judge column name to include.

    Returns:
        pd.DataFrame: One row per (item_id, judge) where an outcome exists.

    Raises:
        ValueError: If required metadata is missing or outcome is unexpected.
    """

    required_meta = {
        "persona",
        "prompt_type",
        "model_1",
        "model_2",
    }
    rows: List[Dict[str, Any]] = []

    for judge, mapping in outcomes_by_judge.items():
        if not isinstance(mapping, dict):
            continue
        j = str(judge)
        for item_id in item_ids:
            key = str(item_id)
            if key not in mapping:
                continue
            meta = item_meta_by_id.get(key) or {}
            if not meta.get("pairwise_judgment_type"):
                meta = dict(meta)
                meta["pairwise_judgment_type"] = "persona"
            missing = [m for m in sorted(required_meta) if not meta.get(m)]
            if missing:
                raise ValueError(
                    "Missing required item metadata for positional win-rate frame. "
                    f"item_id={key!r} missing={missing!r}"
                )

            uid = str(meta["persona"])
            vlabel = str(meta["prompt_type"])
            m1 = str(meta["model_1"])
            m2 = str(meta["model_2"])
            task_id = str(meta.get("base_task_id") or meta.get("task_id") or key)
            variant_id = str(meta.get("variant_id") or "")

            outcome = str(mapping.get(key))
            if outcome == "tie":
                overall = "tie"
            elif outcome == "model_1":
                overall = "model_a"
            elif outcome == "model_2":
                overall = "model_b"
            else:
                raise ValueError(
                    "Unexpected positional outcome value. "
                    f"item_id={key!r} judge={j!r} outcome={outcome!r} expected_one_of={['model_1','model_2','tie']!r}"
                )

            rows.append(
                {
                    "user_id": uid,
                    "variant_label": vlabel,
                    "task_id": task_id,
                    "variant_id": variant_id,
                    "model_a_name": m1,
                    "model_b_name": m2,
                    "pairwise_judgment_type": str(meta["pairwise_judgment_type"]),
                    "overall_winner_label": overall,
                    judge_column: j,
                }
            )

    return pd.DataFrame(rows)


def _ensure_objective_flip_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure objective flip columns exist for stable win-rate table schemas.

    Streamlit does not always have objective metrics loaded, but Stage 6 includes
    these columns in joint preference long tables when available.

    Args:
        frame: Joint preference long table.

    Returns:
        DataFrame: Copy with objective flip columns present (filled with NA if absent).
    """

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for col in [
        "objective_flip_rate_vs_original",
        "objective_reverse_flip_rate_vs_original",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _add_verbose_personalized_mark_column(
    frame: pd.DataFrame, *, paired_alpha: float
) -> pd.DataFrame:
    """
    Add Stage-6-style verbose mark column for personalized win-rate rows.

    The output column is:
      - formatted_win_rate_personalized_vs_baselines

    It is populated for `prompt_type` in ('personalized', 'simple_personalized')
    and left empty otherwise.

    Args:
        frame: Joint preference long table (after paired-stats merge).
        paired_alpha: Significance cutoff.

    Returns:
        DataFrame: Copy with the verbose column attached.
    """

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "formatted_win_rate" not in out.columns or "prompt_type" not in out.columns:
        out["formatted_win_rate_personalized_vs_baselines"] = ""
        return out

    def _to_float(x: object) -> Optional[float]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    alpha = float(paired_alpha)

    def _mark_words(
        delta: object, *, p_value: object = None, q_value: object = None
    ) -> str:
        d = _to_float(delta)
        q = _to_float(q_value)
        p = _to_float(p_value)
        pv = q if q is not None else p
        if d is None or pv is None:
            return ""
        if float(pv) > alpha:
            return ""
        if float(d) > 0:
            return "Higher"
        if float(d) < 0:
            return "Lower"
        return "Equal"

    def _mark_p(delta: object, p_value: object) -> str:
        return _mark_words(delta, p_value=p_value, q_value=None)

    def _mark_q(delta: object, q_value: object) -> str:
        return _mark_words(delta, p_value=None, q_value=q_value)

    def _row_formatted_personalized_vs_baselines(r: pd.Series) -> str:
        pt = str(r.get("prompt_type") or "")
        if pt not in ("personalized", "simple_personalized"):
            return ""

        base = str(r.get("formatted_win_rate") or "").strip()
        if not base:
            return ""

        o_p = _mark_p(
            r.get("paired_treat_vs_orig_delta_win_mean"),
            r.get("paired_treat_vs_orig_delta_win_p_value"),
        )
        o_q = _mark_q(
            r.get("paired_treat_vs_orig_delta_win_mean"),
            r.get("paired_treat_vs_orig_delta_win_q_value"),
        )
        c_p = _mark_p(
            r.get("paired_treat_vs_ctrl_delta_win_mean"),
            r.get("paired_treat_vs_ctrl_delta_win_p_value"),
        )
        c_q = _mark_q(
            r.get("paired_treat_vs_ctrl_delta_win_mean"),
            r.get("paired_treat_vs_ctrl_delta_win_q_value"),
        )
        return f"{base} o(p:{o_p},q:{o_q}) c(p:{c_p},q:{c_q})"

    out["formatted_win_rate_personalized_vs_baselines"] = out.apply(
        _row_formatted_personalized_vs_baselines, axis=1
    )
    return out


def _maybe_attach_paired_stats(
    *,
    sample_df: pd.DataFrame,
    long_df: pd.DataFrame,
    group_by_judge: bool,
    judge_column: str,
    paired_alpha: float,
) -> pd.DataFrame:
    """
    Attach paired-test columns and the verbose personalized mark column.

    Args:
        sample_df: Pairwise sample-level DataFrame (Stage-6-shaped).
        long_df: Joint preference long table (pooled or by-judge).
        group_by_judge: If True, compute paired stats per judge.
        judge_column: Judge column name.
        paired_alpha: Significance cutoff.

    Returns:
        DataFrame: Augmented long table.
    """

    if long_df is None or long_df.empty:
        return long_df
    out = _ensure_objective_flip_columns(long_df)

    if "formatted_win_rate_personalized_vs_baselines" not in out.columns:
        out["formatted_win_rate_personalized_vs_baselines"] = ""

    if sample_df is None or sample_df.empty:
        return out

    required = {
        "user_id",
        "task_id",
        "variant_label",
        "variant_id",
        "model_a_name",
        "model_b_name",
        "overall_winner_label",
    }
    missing = sorted([c for c in required if c not in sample_df.columns])
    if missing:
        return out

    variants = set(sample_df["variant_label"].astype(str).unique().tolist())
    if not ({"original", "personalized"} <= variants):
        return out

    paired = compute_cluster_aware_paired_tests_for_joint_preference(
        sample_df,
        n_permutations=10_000,
        n_bootstrap=2_000,
        alpha=float(paired_alpha),
        seed=1,
        apply_bh_fdr=True,
        group_by_judge=bool(group_by_judge),
        judge_column=str(judge_column),
    )
    if paired is None or paired.empty:
        return out

    join_cols = ["persona", "row_model", "col_model"]
    if (
        group_by_judge
        and "judge" in out.columns
        and str(judge_column) in paired.columns
    ):
        join_cols = join_cols + ["judge"]

    out = out.merge(paired, on=join_cols, how="left")
    out = _add_verbose_personalized_mark_column(out, paired_alpha=float(paired_alpha))
    return out


def _render_dimension_details(
    judge: str,
    task_id: str,
    dimension_name: str,
    dim_result: Dict[str, Any],
) -> None:
    """
    Render one dimension's judgment details.

    Args:
        judge: Judge model name.
        task_id: Task identifier for keying.
        dimension_name: Dimension key.
        dim_result: Dimension result payload.
    """

    winner = str(dim_result.get("winner", "tie"))
    confidence = str(dim_result.get("confidence", "low"))
    bias = bool(dim_result.get("position_bias_detected", False))
    winner_model = dim_result.get("winner_model")

    st.markdown(
        f"**winner**: `{winner}`"
        + (f" (model={winner_model})" if winner_model else "")
        + f"  \n**confidence**: `{confidence}`  \n**position_bias_detected**: `{bias}`"
    )

    rationale = str(dim_result.get("rationale", "") or "")
    raw = str(dim_result.get("raw_response", "") or "")

    if rationale:
        _render_text_block(
            "rationale",
            rationale,
            height=140,
            key=f"ta_rationale_{judge}_{task_id}_{dimension_name}",
        )
    if raw:
        _render_text_block(
            "raw_response",
            raw,
            height=220,
            key=f"ta_raw_{judge}_{task_id}_{dimension_name}",
        )

    original_order = dim_result.get("original_order_result") or {}
    swapped_order = dim_result.get("swapped_order_result") or {}
    if isinstance(original_order, dict) and original_order:
        with st.expander("original_order_result", expanded=False):
            st.json(original_order)
    if isinstance(swapped_order, dict) and swapped_order:
        with st.expander("swapped_order_result", expanded=False):
            st.json(swapped_order)


def _winner_display_name(
    *,
    winner_model: Optional[object],
    winner_value: Optional[object],
    model_a_name: str,
    model_b_name: str,
) -> str:
    """
    Convert winner fields into a display-friendly model name.

    Args:
        winner_model: Optional explicit winner model name (preferred when present).
        winner_value: Winner label ('A', 'B', or 'tie') when winner_model is missing.
        model_a_name: Name of model A for this comparison.
        model_b_name: Name of model B for this comparison.

    Returns:
        str: Winner model name or 'tie'.
    """

    if winner_model is not None and str(winner_model).strip():
        return str(winner_model)
    val = str(winner_value or "tie").strip()
    if val == "A":
        return model_a_name
    if val == "B":
        return model_b_name
    return "tie"


def _confidence_rank(value: object) -> int:
    """
    Map confidence label to an ordinal rank.

    Args:
        value: Confidence value (string-like).

    Returns:
        int: Rank where higher is more confident.
    """

    text = str(value or "low").strip().lower()
    if text == "high":
        return 3
    if text == "medium":
        return 2
    return 1


def _recompute_dimension_winner(
    *,
    dim_result: Dict[str, Any],
    mode: str,
    model_a_name: str,
    model_b_name: str,
) -> Tuple[str, str]:
    """
    Recompute (winner_name, confidence) for a single dimension.

    Modes:
    - stored: use stored winner fields (winner_model/winner).
    - strict: if original/swapped order results exist, require agreement; otherwise tie.
    - finegrained: if original/swapped disagree, break ties using confidence ranks (Stage-6-style).

    Args:
        dim_result: Dimension result payload from Stage 5b.
        mode: One of {'stored','strict','finegrained'}.
        model_a_name: Model A name (from record).
        model_b_name: Model B name (from record).

    Returns:
        Tuple[str, str]: (winner_name_or_tie, confidence_label)
    """

    mode_token = str(mode or "stored").strip().lower()
    if mode_token not in {"stored", "strict", "finegrained"}:
        raise ValueError(f"Unknown tie-breaker mode: {mode!r}")

    # Stored mode: use the record as-is.
    if mode_token == "stored":
        winner_model = dim_result.get("winner_model")
        winner_value = dim_result.get("winner")
        winner_name = _winner_display_name(
            winner_model=winner_model,
            winner_value=winner_value,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        confidence = str(dim_result.get("confidence", "low"))
        return winner_name, confidence

    original_order = dim_result.get("original_order_result")
    swapped_order = dim_result.get("swapped_order_result")
    if not isinstance(original_order, dict) or not isinstance(swapped_order, dict):
        # Without both orders, we cannot recompute; fall back to stored.
        winner_model = dim_result.get("winner_model")
        winner_value = dim_result.get("winner")
        winner_name = _winner_display_name(
            winner_model=winner_model,
            winner_value=winner_value,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        confidence = str(dim_result.get("confidence", "low"))
        return winner_name, confidence

    orig_pos_winner = str(original_order.get("position_winner", "tie"))
    swapped_pos_winner = str(swapped_order.get("position_winner", "tie"))
    orig_conf = str(original_order.get("confidence", "low"))
    swapped_conf = str(swapped_order.get("confidence", "low"))

    # Convert position winners to model winners.
    # Original order: position A = model_a, position B = model_b
    if orig_pos_winner == "A":
        orig_model_winner: Optional[str] = model_a_name
    elif orig_pos_winner == "B":
        orig_model_winner = model_b_name
    else:
        orig_model_winner = None

    # Swapped order: position A = model_b, position B = model_a
    if swapped_pos_winner == "A":
        swapped_model_winner: Optional[str] = model_b_name
    elif swapped_pos_winner == "B":
        swapped_model_winner = model_a_name
    else:
        swapped_model_winner = None

    # Agreement case.
    if orig_model_winner == swapped_model_winner:
        winner_name = orig_model_winner or "tie"
        return winner_name, orig_conf

    # Disagreement: strict mode always ties.
    if mode_token == "strict":
        return "tie", "low"

    # Fine-grained tie breaker (mirrors Stage 6 logic).
    orig_rank = _confidence_rank(orig_conf)
    swapped_rank = _confidence_rank(swapped_conf)

    # One order confident winner (>= medium) and the other is tie.
    if (
        orig_model_winner is not None
        and orig_rank >= 2
        and swapped_model_winner is None
    ):
        return orig_model_winner, orig_conf
    if (
        swapped_model_winner is not None
        and swapped_rank >= 2
        and orig_model_winner is None
    ):
        return swapped_model_winner, swapped_conf

    # Both ties.
    if orig_model_winner is None and swapped_model_winner is None:
        return "tie", "low"

    # One tie but other low-confidence winner: remain tie.
    if orig_model_winner is None or swapped_model_winner is None:
        return "tie", "low"

    # Both winners: pick higher confidence, else tie.
    if orig_rank > swapped_rank:
        return orig_model_winner, orig_conf
    if swapped_rank > orig_rank:
        return swapped_model_winner, swapped_conf
    return "tie", "low"


def _recompute_overall_from_dimensions(
    *,
    record: Dict[str, Any],
    mode: str,
) -> Tuple[str, int, int, int]:
    """
    Recompute (overall_winner, a_wins, b_wins, ties) for a record.

    Args:
        record: Pairwise record.
        mode: Tie-breaker mode.

    Returns:
        Tuple[str, int, int, int]: (winner_name_or_tie, model_a_wins, model_b_wins, ties)
    """

    overall, a, b, t, _valid = recompute_overall_from_dimensions_with_controls(
        record=record,
        mode=mode,
        omit_pairwise_keys=(),
        exclude_eval_parse_errors=False,
    )
    return overall, a, b, t


def _render_compact_judge_summary(
    *,
    judges: List[str],
    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
    task_id: str,
    model_a_name: str,
    model_b_name: str,
    tie_breaker_mode: str,
    omit_pairwise_keys: Sequence[str],
    exclude_eval_parse_errors: bool,
    drop_invalid_samples: bool,
    dimension_weighted_overall_winner: bool,
    dimension_weights_by_user: Optional[Dict[str, Dict[str, float]]],
    correctness_mode: str = "ignore",
    objective_lookup: Optional[Dict] = None,
    include_plus_correctness: bool = False,
    use_human_overall_choice: bool = False,
) -> None:
    """
    Render a compact overall-winner summary for all judges.

    Args:
        judges: Ordered list of judge names.
        index_by_judge: Judge -> task_id -> record.
        task_id: Current task identifier.
        model_a_name: Model A name.
        model_b_name: Model B name.
        tie_breaker_mode: Tie-breaker mode.
        omit_pairwise_keys: Dimension keys to omit.
        exclude_eval_parse_errors: If True, exclude error dimensions.
        drop_invalid_samples: If True, mark invalid samples.
        dimension_weighted_overall_winner: If True, use dimension-weighted voting.
        dimension_weights_by_user: Persona dimension weights.
        correctness_mode: Correctness handling mode.
        objective_lookup: Pre-built objective lookup.
        include_plus_correctness: If True, include plus correctness.
        use_human_overall_choice: If True, human-judge rows use direct
            ``human_overall_choice`` and skip recomputation.
    """

    st.markdown("### Overall winner")
    rows: List[Dict[str, Any]] = []
    for judge in judges:
        record = index_by_judge.get(judge, {}).get(task_id)
        if record is None:
            continue

        record_model_a = str(record.get("model_a_name", model_a_name))
        record_model_b = str(record.get("model_b_name", model_b_name))

        direct_human_winner = None
        if bool(use_human_overall_choice):
            direct_human_winner = _resolve_direct_human_overall_winner(
                record, judge_token=str(judge)
            )
        if direct_human_winner is not None:
            if direct_human_winner == "tie":
                winner = "tie"
            elif direct_human_winner == record_model_a:
                winner = (
                    model_a_name if record_model_a == model_a_name else model_b_name
                )
            elif direct_human_winner == record_model_b:
                winner = (
                    model_a_name if record_model_b == model_a_name else model_b_name
                )
            else:
                winner = str(direct_human_winner)

            rows.append(
                {
                    "Judge": judge,
                    "Winner": winner,
                    f"{model_a_name} Wins": None,
                    f"{model_b_name} Wins": None,
                    "Ties": None,
                    "weighted_score_a": None,
                    "weighted_score_b": None,
                    "weighted_score_tie": None,
                    "valid_for_overall": True,
                    "overall_source": "human_overall_choice",
                }
            )
            continue

        corr_a, corr_b, plus_a, plus_b = _get_record_correctness(
            record, correctness_mode, objective_lookup
        )

        # Always compute unweighted counts for transparency.
        (
            _overall_unweighted,
            recomputed_a,
            recomputed_b,
            recomputed_ties,
            _valid_unweighted,
        ) = recompute_overall_from_dimensions_with_controls(
            record=record,
            mode=tie_breaker_mode,
            omit_pairwise_keys=omit_pairwise_keys,
            exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
            correctness_mode=correctness_mode,
            correctness_a=corr_a,
            correctness_b=corr_b,
            plus_correctness_a=plus_a,
            plus_correctness_b=plus_b,
            include_plus_correctness=include_plus_correctness,
        )

        score_a: Optional[float] = None
        score_b: Optional[float] = None
        score_tie: Optional[float] = None
        if bool(dimension_weighted_overall_winner):
            if dimension_weights_by_user is None:
                raise ValueError(
                    "dimension_weights_by_user is required when "
                    "dimension_weighted_overall_winner is enabled."
                )
            (
                recomputed_overall,
                score_a,
                score_b,
                score_tie,
                valid_for_overall,
            ) = recompute_overall_from_dimensions_with_controls_weighted(
                record=record,
                mode=tie_breaker_mode,
                omit_pairwise_keys=omit_pairwise_keys,
                exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                dimension_weights_by_user=dimension_weights_by_user,
                correctness_mode=correctness_mode,
                correctness_a=corr_a,
                correctness_b=corr_b,
                plus_correctness_a=plus_a,
                plus_correctness_b=plus_b,
                include_plus_correctness=include_plus_correctness,
            )
        else:
            (
                recomputed_overall,
                _a,
                _b,
                _t,
                valid_for_overall,
            ) = recompute_overall_from_dimensions_with_controls(
                record=record,
                mode=tie_breaker_mode,
                omit_pairwise_keys=omit_pairwise_keys,
                exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                correctness_mode=correctness_mode,
                correctness_a=corr_a,
                correctness_b=corr_b,
                plus_correctness_a=plus_a,
                plus_correctness_b=plus_b,
                include_plus_correctness=include_plus_correctness,
            )

        # Map record-space counts onto canonical display model names.
        a_wins = 0
        b_wins = 0
        if record_model_a == model_a_name:
            a_wins += recomputed_a
        elif record_model_a == model_b_name:
            b_wins += recomputed_a
        if record_model_b == model_a_name:
            a_wins += recomputed_b
        elif record_model_b == model_b_name:
            b_wins += recomputed_b

        winner = "tie"
        if not bool(valid_for_overall) and bool(drop_invalid_samples):
            winner = "invalid"
        elif recomputed_overall == record_model_a:
            winner = model_a_name if record_model_a == model_a_name else model_b_name
        elif recomputed_overall == record_model_b:
            winner = model_a_name if record_model_b == model_a_name else model_b_name
        elif recomputed_overall == "tie":
            winner = "tie"
        else:
            winner = str(recomputed_overall)

        rows.append(
            {
                "Judge": judge,
                "Winner": winner,
                f"{model_a_name} Wins": a_wins,
                f"{model_b_name} Wins": b_wins,
                "Ties": int(recomputed_ties),
                "weighted_score_a": score_a,
                "weighted_score_b": score_b,
                "weighted_score_tie": score_tie,
                "valid_for_overall": bool(valid_for_overall),
                "overall_source": "recomputed_dimensions",
            }
        )

    if not rows:
        st.warning("No judge summaries available for this sample.")
        return

    st.dataframe(
        rows,
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _build_outcomes_by_judge_for_task_ids(
    *,
    judges: Sequence[str],
    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
    task_ids: Sequence[str],
    tie_breaker_mode: str,
    model_x_name: str,
    model_y_name: str,
    omit_pairwise_keys: Sequence[str],
    exclude_eval_parse_errors: bool,
    drop_invalid_samples: bool,
    dimension_weighted_overall_winner: bool,
    dimension_weights_by_user: Optional[Dict[str, Dict[str, float]]],
    correctness_mode: str = "ignore",
    objective_lookup: Optional[Dict] = None,
    include_plus_correctness: bool = False,
    use_human_overall_choice: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Build judge -> task_id -> outcome mapping for overall winners.

    Outcomes are normalized to {model_x_name, model_y_name, 'tie'}.

    Args:
        judges: Judge names.
        index_by_judge: Judge -> task_id -> record.
        task_ids: Task ids to include.
        tie_breaker_mode: stored/strict/finegrained.
        model_x_name: Display/canonical model name for X (from sample).
        model_y_name: Display/canonical model name for Y (from sample).
        correctness_mode: Correctness handling mode.
        objective_lookup: Pre-built objective lookup.
        include_plus_correctness: If True, include plus correctness.
        use_human_overall_choice: If True, human-judge rows use direct
            ``human_overall_choice`` and skip recomputation.

    Returns:
        Mapping judge -> (task_id -> outcome).
    """

    out: Dict[str, Dict[str, str]] = {}
    for judge in judges:
        by_task: Dict[str, str] = {}
        indexed = index_by_judge.get(judge, {})
        for tid in task_ids:
            record = indexed.get(tid)
            if record is None:
                continue
            record_model_a = str(record.get("model_a_name", model_x_name))
            record_model_b = str(record.get("model_b_name", model_y_name))

            direct_human_winner = None
            if bool(use_human_overall_choice):
                direct_human_winner = _resolve_direct_human_overall_winner(
                    record, judge_token=str(judge)
                )
            if direct_human_winner is not None:
                if direct_human_winner == "tie":
                    by_task[tid] = "tie"
                elif direct_human_winner == record_model_a:
                    by_task[tid] = (
                        model_x_name if record_model_a == model_x_name else model_y_name
                    )
                elif direct_human_winner == record_model_b:
                    by_task[tid] = (
                        model_x_name if record_model_b == model_x_name else model_y_name
                    )
                elif str(direct_human_winner) == model_x_name:
                    by_task[tid] = model_x_name
                elif str(direct_human_winner) == model_y_name:
                    by_task[tid] = model_y_name
                else:
                    raise ValueError(
                        "Direct human overall choice could not be mapped to display models. "
                        f"judge={judge!r} task_id={tid!r} direct_human_winner={direct_human_winner!r} "
                        f"model_x_name={model_x_name!r} model_y_name={model_y_name!r}"
                    )
                continue

            corr_a, corr_b, plus_a, plus_b = _get_record_correctness(
                record, correctness_mode, objective_lookup
            )

            if bool(dimension_weighted_overall_winner):
                if dimension_weights_by_user is None:
                    raise ValueError(
                        "dimension_weights_by_user is required when "
                        "dimension_weighted_overall_winner is enabled."
                    )
                record_user_id = str(record.get("user_id", "")).strip()
                if record_user_id not in dimension_weights_by_user:
                    logger.error(
                        "missing persona weights during outcome recomputation: judge=%s task_id=%s user_id=%s source_persona=%s available_weight_keys=%s artifact_record_models=(%s,%s)",
                        judge,
                        tid,
                        record_user_id,
                        record.get("source_persona"),
                        sorted(dimension_weights_by_user.keys()),
                        record_model_a,
                        record_model_b,
                    )
                (
                    recomputed_overall,
                    _score_a,
                    _score_b,
                    _score_tie,
                    valid_for_overall,
                ) = recompute_overall_from_dimensions_with_controls_weighted(
                    record=record,
                    mode=tie_breaker_mode,
                    omit_pairwise_keys=omit_pairwise_keys,
                    exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                    dimension_weights_by_user=dimension_weights_by_user,
                    correctness_mode=correctness_mode,
                    correctness_a=corr_a,
                    correctness_b=corr_b,
                    plus_correctness_a=plus_a,
                    plus_correctness_b=plus_b,
                    include_plus_correctness=include_plus_correctness,
                )
            else:
                recomputed_overall, _a, _b, _t, valid_for_overall = (
                    recompute_overall_from_dimensions_with_controls(
                        record=record,
                        mode=tie_breaker_mode,
                        omit_pairwise_keys=omit_pairwise_keys,
                        exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                        correctness_mode=correctness_mode,
                        correctness_a=corr_a,
                        correctness_b=corr_b,
                        plus_correctness_a=plus_a,
                        plus_correctness_b=plus_b,
                        include_plus_correctness=include_plus_correctness,
                    )
                )
            if bool(drop_invalid_samples) and not bool(valid_for_overall):
                continue
            if recomputed_overall == "tie":
                outcome = "tie"
            elif recomputed_overall == record_model_a:
                outcome = (
                    model_x_name if record_model_a == model_x_name else model_y_name
                )
            elif recomputed_overall == record_model_b:
                outcome = (
                    model_x_name if record_model_b == model_x_name else model_y_name
                )
            elif str(recomputed_overall) == model_x_name:
                outcome = model_x_name
            elif str(recomputed_overall) == model_y_name:
                outcome = model_y_name
            else:
                outcome = "tie"
            by_task[tid] = outcome
        out[judge] = by_task
    return out


def _render_dimension_comparison_table(
    *,
    judges: List[str],
    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
    task_id: str,
    model_a_name: str,
    model_b_name: str,
    tie_breaker_mode: str,
    omit_pairwise_keys: Sequence[str],
    exclude_eval_parse_errors: bool,
) -> None:
    """
    Render a per-dimension comparison table across judges.

    Args:
        judges: Ordered list of judge names.
        index_by_judge: Judge -> task_id -> record.
        task_id: Current task identifier.
        model_a_name: Model A name.
        model_b_name: Model B name.
    """

    dim_names: set[str] = set()
    for judge in judges:
        rec = index_by_judge.get(judge, {}).get(task_id)
        if rec is None:
            continue
        dim_results = rec.get("dimension_results") or {}
        if isinstance(dim_results, dict):
            dim_names.update(str(k) for k in dim_results.keys())

    omit_set = {str(x) for x in (omit_pairwise_keys or [])}
    dim_names = {d for d in dim_names if str(d) not in omit_set}

    if not dim_names:
        st.error("No dimension_results found for this sample across judges.")
        return

    rows: List[Dict[str, Any]] = []
    for dim in sorted(dim_names):
        row: Dict[str, Any] = {"Dimension": dim}
        for judge in judges:
            rec = index_by_judge.get(judge, {}).get(task_id)
            if rec is None:
                row[judge] = "missing"
                continue
            dim_results = rec.get("dimension_results") or {}
            dim_payload = (
                dim_results.get(dim) if isinstance(dim_results, dict) else None
            )
            if not isinstance(dim_payload, dict):
                row[judge] = "missing"
                continue

            rec_model_a = str(rec.get("model_a_name", model_a_name))
            rec_model_b = str(rec.get("model_b_name", model_b_name))
            if bool(exclude_eval_parse_errors) and is_eval_or_parse_error_rationale(
                dim_payload.get("rationale", "")
            ):
                row[judge] = "excluded(error)"
                continue
            winner_name, confidence = _recompute_dimension_winner(
                dim_result=dim_payload,
                mode=tie_breaker_mode,
                model_a_name=rec_model_a,
                model_b_name=rec_model_b,
            )

            # Map winner name into canonical display names when possible.
            if winner_name == rec_model_a:
                display_winner = (
                    model_a_name if rec_model_a == model_a_name else model_b_name
                )
            elif winner_name == rec_model_b:
                display_winner = (
                    model_a_name if rec_model_b == model_a_name else model_b_name
                )
            else:
                display_winner = winner_name

            conf_disp = str(confidence or "low").strip().capitalize()
            if display_winner == "tie":
                row[judge] = f"tie (confidence: {conf_disp})"
            else:
                row[judge] = f"{display_winner} (confidence: {conf_disp})"
        rows.append(row)

    st.markdown("### Per-dimension table")
    st.dataframe(
        rows,
        **_width_kwargs(st.dataframe, _ui_width_mode()),
        hide_index=True,
    )


def _render_judge_record(
    judge: str, record: Dict[str, Any], artifact_path: str
) -> None:
    """
    Render a judge's full record for a single sample.

    Args:
        judge: Judge model name.
        record: Pairwise record.
        artifact_path: Path to the artifact file used for this judge.
    """

    with st.expander(f"Judge: {judge}", expanded=True):
        st.caption(f"artifact: `{artifact_path}`")

        task_id = str(record.get("task_id", "unknown_task"))
        overall = record.get("overall_winner")
        st.markdown(f"**overall_winner**: `{overall}`")
        metadata = _record_public_metadata(record)
        if "human_overall_choice" in metadata:
            st.markdown(
                f"**metadata.human_overall_choice**: `{metadata.get('human_overall_choice')}`"
            )

        win_counts = record.get("win_counts")
        if isinstance(win_counts, dict) and win_counts:
            st.json({"win_counts": win_counts})

        dim_results = record.get("dimension_results") or {}
        if not isinstance(dim_results, dict) or not dim_results:
            st.error("dimension_results missing or empty for this judge record.")
            return

        dims_sorted = sorted(dim_results.keys())
        st.markdown(f"**dimensions**: {', '.join(f'`{d}`' for d in dims_sorted)}")
        for dim_name in dims_sorted:
            dim_payload = dim_results.get(dim_name) or {}
            if not isinstance(dim_payload, dict):
                st.error(
                    f"Invalid dimension payload for `{dim_name}` (expected object)."
                )
                continue
            with st.expander(f"dimension: {dim_name}", expanded=False):
                _render_dimension_details(judge, task_id, dim_name, dim_payload)


def _reset_sample_index() -> None:
    """Reset the sample navigation index in session state."""

    st.session_state["sample_idx"] = 0


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(
        page_title="Pairwise Comparison Explorer (Stage 5b)",
        layout="wide",
    )
    st.title("Pairwise Comparison Explorer (Stage 5b)")

    st.sidebar.header("Data source")
    # ------------------------------------------------------------------
    # Sidebar defaults (apply on first load; safe if options missing)
    # ------------------------------------------------------------------
    st.session_state.setdefault("layout_density", "compact")
    st.sidebar.selectbox(
        "Layout",
        options=["compact", "comfortable"],
        index=0,
        key="layout_density",
    )
    run_dir_names = _list_immediate_subdirs(RUNS_ROOT_DIR)
    if not run_dir_names:
        st.sidebar.error(
            f"No run directories found under `{RUNS_ROOT_DIR}`. "
            "Expected subdirectories like `new_gpt5_100_samples/`."
        )
        st.stop()

    default_run_idx = (
        run_dir_names.index(DEFAULT_RUNS_SUBDIR)
        if DEFAULT_RUNS_SUBDIR in run_dir_names
        else 0
    )
    selected_run_dir = st.sidebar.selectbox(
        "Base results directory",
        options=run_dir_names,
        index=default_run_idx,
        key="base_results_dir",
        on_change=_reset_sample_index,
    )
    base_dir_str = str((RUNS_ROOT_DIR / selected_run_dir).resolve())

    if st.sidebar.button(
        "Rescan",
        **_width_kwargs(st.sidebar.button, _ui_width_mode()),
    ):
        _cached_discover_configs.clear()
        _cached_discover_pairs.clear()
        _cached_load_latest_records.clear()
        _cached_load_indexed_records_for_artifact.clear()
        _cached_load_dimension_weights_by_user.clear()
        _cached_load_stage6_pairwise_rows.clear()
        _reset_sample_index()

    try:
        configs_dicts = _cached_discover_configs(base_dir_str)
    except Exception as exc:
        st.error(f"Failed to scan base results directory: {exc}")
        st.stop()

    if not configs_dicts:
        st.warning(
            "No pairwise configurations found. Expected directories like "
            "`<base>/<persona>/5b_pairwise_evaluation/.../pairwise-comparison_*.json`."
        )
        st.stop()

    configs = [PairwiseConfigKey(**d) for d in configs_dicts]

    st.sidebar.header("Configuration")
    persona_options = sorted({c.persona for c in configs})
    desired_persona = "novice_user"
    if (
        "cfg_persona" not in st.session_state
        or st.session_state.get("cfg_persona") not in persona_options
    ):
        st.session_state["cfg_persona"] = (
            desired_persona
            if desired_persona in persona_options
            else persona_options[0]
        )
    selected_persona = st.sidebar.selectbox(
        "persona",
        options=persona_options,
        key="cfg_persona",
        on_change=_reset_sample_index,
    )
    configs_for_persona = [c for c in configs if c.persona == selected_persona]

    gen_options = sorted({c.generator_model for c in configs_for_persona})
    selected_gen = st.sidebar.selectbox(
        "gen_model",
        options=gen_options,
        key="cfg_gen_model",
        on_change=_reset_sample_index,
    )
    configs_for_gen = [
        c for c in configs_for_persona if c.generator_model == selected_gen
    ]

    filter_options = sorted({c.filter_model for c in configs_for_gen})
    selected_filter = st.sidebar.selectbox(
        "filter_model",
        options=filter_options,
        key="cfg_filter_model",
        on_change=_reset_sample_index,
    )
    configs_for_filter = [
        c for c in configs_for_gen if c.filter_model == selected_filter
    ]

    judgment_type_options = _available_judgment_types(configs_for_filter)
    desired_judgment_type = "persona"
    if (
        "cfg_pairwise_judgment_type" not in st.session_state
        or st.session_state.get("cfg_pairwise_judgment_type")
        not in judgment_type_options
    ):
        st.session_state["cfg_pairwise_judgment_type"] = (
            desired_judgment_type
            if desired_judgment_type in judgment_type_options
            else judgment_type_options[0]
        )
    selected_judgment_type = st.sidebar.selectbox(
        "pairwise_judgment_type",
        options=judgment_type_options,
        key="cfg_pairwise_judgment_type",
        on_change=_reset_sample_index,
    )
    configs_for_judgment_type = [
        c
        for c in configs_for_filter
        if str(c.pairwise_judgment_type) == str(selected_judgment_type)
    ]

    prompt_labels = sorted(
        {format_prompt_type(c.prompt_type) for c in configs_for_judgment_type}
    )
    desired_prompt_label = "personalized"
    if (
        "cfg_prompt_type" not in st.session_state
        or st.session_state.get("cfg_prompt_type") not in prompt_labels
    ):
        st.session_state["cfg_prompt_type"] = (
            desired_prompt_label
            if desired_prompt_label in prompt_labels
            else (prompt_labels[0] if prompt_labels else "legacy")
        )
    selected_prompt_label = st.sidebar.selectbox(
        "prompt_type",
        options=prompt_labels,
        key="cfg_prompt_type",
        on_change=_reset_sample_index,
    )

    tie_breaker_labels = {
        "stage6_strict": "stage6_strict (match Stage 6 strict; stored winners)",
        "stage6_finegrained": "stage6_finegrained (match Stage 6 finegrained)",
        "stored": "stored (use artifact winners)",
        "swap_agreement_strict": "swap_agreement_strict (require swap agreement)",
        "finegrained": "finegrained (swap-aware confidence tie-break)",
    }
    desired_tie_breaker = "finegrained"
    if (
        "tie_breaker_mode" not in st.session_state
        or st.session_state.get("tie_breaker_mode") not in tie_breaker_labels
    ):
        st.session_state["tie_breaker_mode"] = (
            desired_tie_breaker
            if desired_tie_breaker in tie_breaker_labels
            else "stored"
        )
    selected_tie_breaker = st.sidebar.selectbox(
        "tie_breaker",
        options=list(tie_breaker_labels.keys()),
        format_func=lambda k: tie_breaker_labels.get(str(k), str(k)),
        key="tie_breaker_mode",
    )
    st.sidebar.caption("Note: Stage 6 strict != swap_agreement_strict.")
    with st.sidebar.expander("Tie-breaker modes (quick explanation)", expanded=False):
        st.markdown(
            "- **stage6_strict**: use stored winners from the artifact (no swap-agreement enforcement).\n"
            "- **swap_agreement_strict**: if original-order vs swapped-order disagree for a dimension, force that dimension to `tie`.\n"
            "- **stage6_finegrained / finegrained**: if original vs swapped disagree, use confidence heuristics; equal low-confidence disagreement becomes `tie`.\n\n"
            "Example (one dimension): original says `A` (low), swapped says `B` (low).\n"
            "- `stage6_strict`: uses stored winner as-is (could still be `A` or `B` depending on what was stored).\n"
            "- `swap_agreement_strict`: becomes `tie`.\n"
            "- `stage6_finegrained`: becomes `tie`.\n"
        )

    st.sidebar.header("Stage 6 recomputation (primary)")
    stage6_tie_breaker = st.sidebar.selectbox(
        "stage6_tie_breaker",
        options=["strict", "finegrained"],
        index=1,
        key="stage6_tie_breaker_mode",
    )
    omit_options = list(PAIRWISE_DIMENSIONS)
    desired_omit_defaults = [
        "friction_loss_of_control",
        "reliability_user_trust",
    ]
    if "stage6_omit_dimensions" not in st.session_state:
        st.session_state["stage6_omit_dimensions"] = desired_omit_defaults
    else:
        # Normalize legacy aliases from older saved session values into canonical dims.
        # Important: preserve an intentionally empty selection.
        existing = st.session_state.get("stage6_omit_dimensions") or []
        st.session_state["stage6_omit_dimensions"] = (
            _canonicalize_omit_dimension_selection([str(x) for x in existing])
        )
    omit_dimensions = st.sidebar.multiselect(
        "omit_dimensions",
        options=omit_options,
        key="stage6_omit_dimensions",
        help=(
            "Omit specific dimensions from Stage-6-style recomputation. "
            "Only canonical dimension names are shown; older legacy aliases are "
            "normalized automatically."
        ),
    )
    exclude_eval_parse_errors = st.sidebar.checkbox(
        "exclude_eval_parse_errors",
        value=True,
        key="stage6_exclude_eval_parse_errors",
        help=(
            "Match Stage 6: exclude dimensions whose rationale starts with "
            "'evaluation error:' or 'parse error:' from overall-winner vote."
        ),
    )
    drop_invalid_samples = st.sidebar.checkbox(
        "drop_invalid_samples",
        value=True,
        key="stage6_drop_invalid_samples",
        help=(
            "Match Stage 6: drop samples with no usable (non-error, non-omitted) dimensions."
        ),
    )
    st.sidebar.caption(
        "Scope: recomputation controls (omit/exclude/drop) affect all views and all tie-breaker modes. "
        "Stage‑6 tables use the Stage‑6 normalized backbone; legacy tables use raw artifacts but apply the same controls."
    )

    st.sidebar.header("Persona-weighted overall winner")
    dimension_weighted_overall_winner = st.sidebar.checkbox(
        "dimension_weighted_overall_winner",
        value=False,
        key="dimension_weighted_overall_winner",
        help=(
            "Match Stage 6: recompute per-sample overall winners using persona-specific "
            "dimension weights derived from persona YAML `output_dimensions`."
        ),
    )

    persona_weight_selection_path = persona_weight_selection_file_for_run_dir(
        Path(base_dir_str)
    )
    try:
        saved_weight_sel = load_persona_weight_config_selection(
            persona_weight_selection_path
        )
    except Exception as exc:
        st.sidebar.error(f"Failed to load persona-weight selection file: {exc}")
        st.sidebar.caption(f"selection file path: `{persona_weight_selection_path}`")
        st.stop()

    aggregate_filter_selection_path = aggregate_filter_selection_file_for_run_dir(
        Path(base_dir_str)
    )
    try:
        saved_aggregate_filters = load_aggregate_filter_selection(
            aggregate_filter_selection_path
        )
    except Exception as exc:
        st.sidebar.error(f"Failed to load aggregate filter selection file: {exc}")
        st.sidebar.caption(f"selection file path: `{aggregate_filter_selection_path}`")
        st.stop()

    persona_weight_catalog = _load_persona_weight_catalog()
    yaml_abs = list(persona_weight_catalog["config_paths"])
    abs_to_display: Dict[str, str] = {}
    display_to_abs: Dict[str, str] = {}
    for abs_p in yaml_abs:
        p = Path(abs_p)
        try:
            disp = str(p.resolve().relative_to(project_root))
        except Exception:
            disp = abs_p
        abs_to_display[abs_p] = disp
        display_to_abs[disp] = abs_p

    stage6_default_aliases = [
        "novice_user",
        "researcher_user",
        "intermediate_learner",
        "advanced_developer",
    ]
    default_abs = []
    seen_default_paths: set[str] = set()
    for alias in stage6_default_aliases:
        cand = persona_weight_catalog["alias_to_config_path"].get(alias)
        if cand and cand not in seen_default_paths:
            seen_default_paths.add(cand)
            default_abs.append(cand)

    saved_abs = [
        str(Path(p).expanduser().resolve()) for p in saved_weight_sel.config_paths
    ]
    valid_saved_abs = [p for p in saved_abs if p in set(yaml_abs)]
    missing_saved = [p for p in saved_abs if p not in set(yaml_abs)]
    if missing_saved:
        st.sidebar.warning(
            "Ignoring saved persona-weight configs outside the canonical experiment "
            f"persona set: {missing_saved!r}"
        )
    desired_abs = valid_saved_abs if valid_saved_abs else default_abs

    desired_display = [abs_to_display[p] for p in desired_abs if p in abs_to_display]
    valid_display_options = [abs_to_display[p] for p in yaml_abs]
    current_selection = (
        st.session_state.get("persona_weight_configs_display")
        if "persona_weight_configs_display" in st.session_state
        else None
    )
    st.session_state["persona_weight_configs_display"] = _normalize_multiselect_state(
        current_selection,
        options=valid_display_options,
        desired_default=desired_display,
    )
    selected_display = st.sidebar.multiselect(
        "persona_weight_configs",
        options=valid_display_options,
        key="persona_weight_configs_display",
        help=(
            "Persona YAML configs used to derive per-dimension weights. "
            "Candidates come from `configs/experiments/components/personas.yaml` "
            "so deprecated legacy profile YAMLs are excluded."
        ),
    )
    selected_weight_yaml_abs = [display_to_abs[d] for d in selected_display]
    st.sidebar.caption(f"Selection file: `{persona_weight_selection_path}`")
    with st.sidebar.expander("Selected persona-weight config paths", expanded=False):
        st.code("\n".join(selected_weight_yaml_abs) if selected_weight_yaml_abs else "")
    if st.sidebar.button(
        "Save persona-weight selection", key="save_persona_weight_sel"
    ):
        try:
            save_persona_weight_config_selection(
                persona_weight_selection_path,
                config_paths=selected_weight_yaml_abs,
                app_version=None,
            )
        except Exception as exc:
            st.sidebar.error(f"Failed to save persona-weight selection: {exc}")
            st.stop()

    persona_weight_selection_mtime = float(
        persona_weight_selection_path.stat().st_mtime
        if persona_weight_selection_path.exists()
        else 0.0
    )

    dimension_weights_by_user: Optional[Dict[str, Dict[str, float]]] = None
    if bool(dimension_weighted_overall_winner):
        try:
            dimension_weights_by_user = _cached_load_dimension_weights_by_user(
                tuple(selected_weight_yaml_abs),
                persona_weight_selection_mtime,
            )
        except Exception as exc:
            st.sidebar.error(f"Failed to load persona dimension weights: {exc}")
            st.stop()
        logger.info(
            "dimension_weighted_overall_winner enabled: selected_persona=%s selected_weight_yaml_abs=%s loaded_weight_keys=%s",
            selected_persona,
            selected_weight_yaml_abs,
            sorted(dimension_weights_by_user.keys()) if dimension_weights_by_user else [],
        )
        selected_weight_rows: List[Dict[str, Any]] = []
        config_path_to_aliases = persona_weight_catalog["config_path_to_aliases"]
        config_path_to_user_id = persona_weight_catalog["config_path_to_user_id"]
        for config_path in selected_weight_yaml_abs:
            selected_weight_rows.append(
                {
                    "config_path": str(config_path),
                    "canonical_user_id": str(
                        config_path_to_user_id.get(str(config_path), "")
                    ),
                    "persona_aliases": ", ".join(
                        sorted(config_path_to_aliases.get(str(config_path), []))
                    ),
                }
            )
        with st.sidebar.expander("Persona-weight diagnostics", expanded=False):
            st.write(
                "Loaded weight keys:",
                (
                    ", ".join(sorted(dimension_weights_by_user.keys()))
                    if dimension_weights_by_user
                    else "(none)"
                ),
            )
            if selected_weight_rows:
                st.dataframe(
                    pd.DataFrame(selected_weight_rows),
                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                    hide_index=True,
                )

    # -- Correctness-aware winner controls --
    st.sidebar.header("Correctness-aware winner")
    correctness_mode = st.sidebar.selectbox(
        "correctness_mode",
        options=["ignore", "dimension", "gate"],
        index=0,
        key="correctness_mode",
        help=(
            "How to incorporate pass@1 correctness into per-sample pairwise winner "
            "determination. 'ignore': no change. 'dimension': treat correctness as "
            "additional vote (weight=5 when dimension-weighted). 'gate': correct model "
            "wins automatically when the other is incorrect."
        ),
    )
    include_plus_correctness = st.sidebar.checkbox(
        "include_plus_correctness",
        value=False,
        key="include_plus_correctness",
        help=(
            "When correctness_mode=dimension, add plus-test pass@1 as an additional "
            "correctness dimension. Its weight equals the persona's workflow_fit weight."
        ),
    )
    use_human_overall_choice = False

    # Build objective lookup for correctness-aware recomputation
    streamlit_objective_lookup: Optional[Dict] = None
    if correctness_mode != "ignore":
        streamlit_objective_lookup = _cached_load_objective_lookup(
            base_dir_str,
        )
        if streamlit_objective_lookup is None or not streamlit_objective_lookup:
            st.sidebar.warning(
                f"Correctness mode '{correctness_mode}' requires objective results, "
                "but no Stage 4 objective data was found under the results directory. "
                "Falling back to 'ignore'."
            )
            correctness_mode = "ignore"
        else:
            st.sidebar.caption(
                f"Loaded {len(streamlit_objective_lookup)} objective lookup entries."
            )

    try:
        config = _select_exact_pairwise_config(
            configs=configs_for_judgment_type,
            prompt_label=selected_prompt_label,
            judgment_type=selected_judgment_type,
        )
    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()
    logger.info(
        "resolved single-view config: base_dir=%s persona=%s generator_model=%s filter_model=%s prompt_type=%s pairwise_judgment_type=%s weighted=%s",
        base_dir_str,
        config.persona,
        config.generator_model,
        config.filter_model,
        format_prompt_type(config.prompt_type),
        config.pairwise_judgment_type,
        bool(dimension_weighted_overall_winner),
    )
    st.sidebar.caption(
        f"Active judgment type: `{config.pairwise_judgment_type}`"
    )
    if (
        bool(dimension_weighted_overall_winner)
        and dimension_weights_by_user is not None
    ):
        expected_user_id = persona_weight_catalog["alias_to_user_id"].get(
            str(config.persona), str(config.persona)
        )
        if expected_user_id not in dimension_weights_by_user:
            st.sidebar.error(
                "Selected persona-weight configs do not cover the current single-view "
                f"persona. persona_alias={config.persona!r} "
                f"expected_user_id={expected_user_id!r} "
                f"loaded_keys={sorted(dimension_weights_by_user.keys())!r}"
            )
            st.stop()

    try:
        pairs_dicts = _cached_discover_pairs(base_dir_str, asdict(config))
    except Exception as exc:
        st.error(f"Failed to discover model pairs for config: {exc}")
        st.stop()

    if not pairs_dicts:
        st.warning("No model pairs found for the selected configuration.")
        st.stop()

    pairs = [ModelPairKey(**p) for p in pairs_dicts]
    pair_labels = [p.label for p in pairs]

    st.sidebar.header("Model pair")
    pair_idx = st.sidebar.selectbox(
        "Select model pair",
        options=list(range(len(pairs))),
        format_func=lambda i: pair_labels[i],
        on_change=_reset_sample_index,
        key="pair_idx",
    )
    pair = pairs[int(pair_idx)]

    base_dir = Path(base_dir_str).expanduser().resolve()
    judges = discover_judges_for_pair(base_dir, config, pair)
    if not judges:
        st.error(
            "No judge outputs found for the selected configuration and model pair."
        )
        st.stop()

    # ------------------------------------------------------------------
    # Artifact selection (persisted per run directory)
    # ------------------------------------------------------------------
    selection_path = selection_file_for_run_dir(Path(base_dir_str))
    try:
        selections_by_key = load_pairwise_artifact_selections(selection_path)
    except Exception as exc:
        st.sidebar.error(f"Failed to load artifact selection file: {exc}")
        st.sidebar.caption(f"selection file path: `{selection_path}`")
        st.stop()

    st.sidebar.header("Artifacts")
    st.sidebar.caption(f"Selection file: `{selection_path}`")

    prompt_label = format_prompt_type(config.prompt_type)
    selected_artifact_by_judge: Dict[str, str] = {}
    stage6_default_by_judge: Dict[str, str] = {}

    for judge in judges:
        artifacts = list_pairwise_artifacts_for_pair_and_judge(
            base_dir, config, pair, judge
        )
        if not artifacts:
            continue

        stage6_default = choose_stage6_default_artifact(artifacts)
        stage6_default_str = str(stage6_default)
        stage6_default_by_judge[str(judge)] = stage6_default_str

        key = PairwiseArtifactSelectionKey(
            persona=str(config.persona),
            generator_model=str(config.generator_model),
            filter_model=str(config.filter_model),
            prompt_type=str(prompt_label),
            pairwise_judgment_type=str(config.pairwise_judgment_type),
            model_a=str(pair.model_a),
            model_b=str(pair.model_b),
            judge_model=str(judge),
        )
        saved_selected = get_saved_selected_path(selections_by_key, key)

        # Use string paths for Streamlit widgets.
        options = [str(p) for p in artifacts]
        options_set = set(options)

        default_path = stage6_default_str
        if saved_selected is not None:
            saved_resolved = str(Path(saved_selected).expanduser().resolve())
            if saved_resolved not in options_set:
                st.sidebar.error(
                    "Saved artifact selection does not match current candidates. "
                    f"judge=`{judge}` saved=`{saved_resolved}`"
                )
                ignore = st.sidebar.checkbox(
                    f"Ignore saved selection for `{judge}` (use Stage6 default)",
                    value=False,
                    key=f"ignore_saved_artifact_{key.to_string()}",
                )
                if not ignore:
                    st.stop()
            else:
                default_path = saved_resolved

        default_idx = options.index(default_path) if default_path in options else 0

        widget_key = f"artifact_select|{key.to_string()}"
        selected = st.sidebar.selectbox(
            f"Artifact for judge `{judge}`",
            options=options,
            index=default_idx,
            format_func=lambda p: format_artifact_option_label(
                Path(p), stage6_default_path=stage6_default
            ),
            key=widget_key,
            on_change=_reset_sample_index,
        )
        st.sidebar.text_area(
            f"Full path ({judge})",
            value=str(selected),
            height=80,
            disabled=True,
            key=f"artifact_path_view|{key.to_string()}",
        )
        selected_artifact_by_judge[str(judge)] = str(selected)

    if not selected_artifact_by_judge:
        st.error("No readable pairwise artifacts found for the selected scope.")
        st.stop()

    if st.sidebar.button(
        "Save artifact selections",
        **_width_kwargs(st.sidebar.button, _ui_width_mode()),
    ):
        updated = dict(selections_by_key)
        for judge, selected_path in selected_artifact_by_judge.items():
            stage6_default_path = stage6_default_by_judge.get(judge)
            if not stage6_default_path:
                continue
            k = PairwiseArtifactSelectionKey(
                persona=str(config.persona),
                generator_model=str(config.generator_model),
                filter_model=str(config.filter_model),
                prompt_type=str(prompt_label),
                pairwise_judgment_type=str(config.pairwise_judgment_type),
                model_a=str(pair.model_a),
                model_b=str(pair.model_b),
                judge_model=str(judge),
            )
            updated = upsert_selection_record(
                updated,
                key=k,
                selected_path=str(selected_path),
                stage6_default_path=str(stage6_default_path),
            )
        try:
            save_pairwise_artifact_selections(selection_path, selections_by_key=updated)
            st.sidebar.success("Saved artifact selections.")
            selections_by_key = updated
        except Exception as exc:
            st.sidebar.error(f"Failed to save artifact selections: {exc}")
            st.stop()

    # Load records from the selected artifacts.
    index_by_judge: Dict[str, Dict[str, Dict[str, Any]]] = {}
    artifact_by_judge: Dict[str, str] = {}
    for judge, artifact_path_str in selected_artifact_by_judge.items():
        try:
            indexed = _cached_load_indexed_records_for_artifact(
                artifact_path_str,
                base_dir_str,
            )
        except Exception as exc:
            _emit_pairwise_load_failure_to_ui(
                exc=exc,
                prefix=f"Failed to load artifact for judge `{judge}`",
                stop_after=True,
            )
        sample_user_ids = sorted(
            {
                str(record.get("user_id", "")).strip()
                for record in indexed.values()
                if str(record.get("user_id", "")).strip()
            }
        )
        logger.info(
            "loaded selected artifact: judge=%s artifact=%s task_count=%d user_ids=%s",
            judge,
            artifact_path_str,
            len(indexed),
            sample_user_ids,
        )
        index_by_judge[str(judge)] = indexed
        artifact_by_judge[str(judge)] = str(artifact_path_str)

    task_ids = build_task_id_union(index_by_judge)
    if not task_ids:
        st.error("No task_ids could be built from the loaded judge artifacts.")
        st.stop()

    # ------------------------------------------------------------------
    # Shared Stage-6 backbone (selected artifacts + Stage-6 sanitization)
    # ------------------------------------------------------------------
    omit_pairwise_keys_stage6, _omit_subjective_stage6 = normalize_omit_dimensions(
        omit_dimensions
    )
    stage6_omit_pairwise_keys_tuple = tuple(
        sorted(str(x) for x in omit_pairwise_keys_stage6)
    )

    stage6_df_selected: pd.DataFrame = pd.DataFrame()
    stage6_backbone_error: Optional[Exception] = None
    try:
        stage6_rows_selected: List[Dict[str, Any]] = []
        for judge_token in sorted(artifact_by_judge.keys(), key=str):
            art = artifact_by_judge.get(judge_token)
            if not art:
                continue
            stage6_rows_selected.extend(
                _cached_load_stage6_pairwise_rows(
                    art,
                    tie_breaker_mode=str(stage6_tie_breaker),
                    omit_pairwise_keys=stage6_omit_pairwise_keys_tuple,
                    exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                    drop_invalid_samples=bool(drop_invalid_samples),
                    judge_token=str(judge_token),
                    dimension_weighted_winner=bool(dimension_weighted_overall_winner),
                    dimension_weight_config_paths=tuple(selected_weight_yaml_abs),
                    dimension_weight_selection_mtime=float(
                        persona_weight_selection_mtime
                    ),
                    correctness_mode=correctness_mode,
                    base_dir_for_objective=base_dir_str,
                    include_plus_correctness=bool(include_plus_correctness),
                    use_human_overall_choice=bool(use_human_overall_choice),
                )
            )
        stage6_df_selected = pd.DataFrame(stage6_rows_selected)
    except Exception as exc:
        stage6_backbone_error = exc

    tabs = st.tabs(["All samples", "Sample"])

    with tabs[1]:
        st.subheader("Available judges")
        st.write(", ".join(f"`{j}`" for j in sorted(index_by_judge.keys())))

        # Build an overall-winners table (same recomputation controls) to pick samples.
        judges_all = sorted(index_by_judge.keys(), key=str)
        outcomes_for_picker = _build_outcomes_by_judge_for_task_ids(
            judges=judges_all,
            index_by_judge=index_by_judge,
            task_ids=list(task_ids),
            tie_breaker_mode=_dimension_tie_breaker_mode(selected_tie_breaker),
            model_x_name=str(pair.model_a),
            model_y_name=str(pair.model_b),
            omit_pairwise_keys=tuple(sorted(str(x) for x in omit_pairwise_keys_stage6)),
            exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
            drop_invalid_samples=bool(drop_invalid_samples),
            dimension_weighted_overall_winner=bool(dimension_weighted_overall_winner),
            dimension_weights_by_user=dimension_weights_by_user,
            correctness_mode=correctness_mode,
            objective_lookup=streamlit_objective_lookup,
            include_plus_correctness=bool(include_plus_correctness),
            use_human_overall_choice=bool(use_human_overall_choice),
        )
        picker_ids: set[str] = set()
        for by_task in outcomes_for_picker.values():
            picker_ids.update(str(k) for k in by_task.keys())
        task_ids_effective = sorted(picker_ids, key=str)

        if not task_ids_effective:
            st.warning(
                "No samples remain under the current recomputation controls. "
                "Try disabling `drop_invalid_samples` or relaxing omit/exclude settings."
            )
            st.stop()

        picker_filters = st.columns(3 if _ui_width_mode() == "stretch" else 2)
        with picker_filters[0]:
            sample_task_contains = st.text_input(
                "task_id contains",
                value="",
                key="sample_flt_task_contains",
            )
        with picker_filters[1]:
            sample_min_disagree = st.number_input(
                "min_disagreeing",
                min_value=0,
                max_value=max(0, len(judges_all) - 1),
                value=0,
                step=1,
                key="sample_flt_min_disagree",
            )
        majority_filter = "Any"
        if _ui_width_mode() == "stretch":
            with picker_filters[2]:
                majority_filter = st.selectbox(
                    "majority_winner",
                    options=["Any", str(pair.model_a), str(pair.model_b), "tie"],
                    index=0,
                    key="sample_flt_majority",
                )

        winners_df = build_overall_winner_table(
            task_ids=task_ids_effective,
            judges=judges_all,
            outcomes_by_judge=outcomes_for_picker,
        )
        view = winners_df.copy()
        if str(sample_task_contains or "").strip():
            needle = str(sample_task_contains).strip()
            view = view[
                view["task_id"].astype(str).str.contains(needle, na=False)
            ].copy()
        if int(sample_min_disagree) > 0 and "n_disagreeing" in view.columns:
            view = view[view["n_disagreeing"] >= int(sample_min_disagree)].copy()
        if str(majority_filter) != "Any" and "majority_winner" in view.columns:
            view = view[
                view["majority_winner"].astype(str) == str(majority_filter)
            ].copy()

        view = view.reset_index(drop=True)
        task_ids_effective_filtered = view["task_id"].astype(str).tolist()
        st.caption(f"Samples in picker: {len(task_ids_effective_filtered)}")

        if not task_ids_effective_filtered:
            st.warning("No samples match the current picker filters.")
            st.stop()

        st.markdown("### Overall winners table (click a row to view)")
        if _dataframe_supports_row_selection():
            sel = st.dataframe(
                view,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
                key="sample_winners_table",
                on_select="rerun",
                selection_mode="single-row",
            )
            try:
                selected_rows = (
                    sel.selection.rows
                    if sel is not None and hasattr(sel, "selection")
                    else []
                )
                if selected_rows:
                    st.session_state["sample_idx"] = int(selected_rows[0])
            except Exception:
                # If selection state is not available in this Streamlit build, ignore.
                pass
        else:
            # Keep the jump selector aligned to the current index.
            cur_idx_for_jump = int(st.session_state.get("sample_idx", 0) or 0)
            cur_idx_for_jump = max(
                0, min(cur_idx_for_jump, len(task_ids_effective_filtered) - 1)
            )
            jump = st.selectbox(
                "Jump to sample",
                options=task_ids_effective_filtered,
                index=cur_idx_for_jump,
                key="sample_jump_task_id",
            )
            if str(jump) in task_ids_effective_filtered:
                st.session_state["sample_idx"] = int(
                    task_ids_effective_filtered.index(str(jump))
                )

        # Navigation (over the filtered picker list)
        if "sample_idx" not in st.session_state:
            st.session_state["sample_idx"] = 0
        max_idx = max(0, len(task_ids_effective_filtered) - 1)
        st.session_state["sample_idx"] = max(
            0, min(int(st.session_state["sample_idx"]), max_idx)
        )

        nav_cols = st.columns(
            [1, 1, 1, 1] if _ui_width_mode() == "content" else [1, 1, 2, 2]
        )
        with nav_cols[0]:
            if st.button(
                "Prev",
                **_width_kwargs(st.button, _ui_width_mode()),
                disabled=st.session_state["sample_idx"] <= 0,
            ):
                st.session_state["sample_idx"] -= 1
        with nav_cols[1]:
            if st.button(
                "Next",
                **_width_kwargs(st.button, _ui_width_mode()),
                disabled=st.session_state["sample_idx"] >= max_idx,
            ):
                st.session_state["sample_idx"] += 1
        with nav_cols[2]:
            st.number_input(
                "Sample index",
                min_value=0,
                max_value=max_idx,
                step=1,
                key="sample_idx",
                label_visibility=(
                    "collapsed" if _ui_width_mode() == "content" else "visible"
                ),
            )
        with nav_cols[3]:
            if _ui_width_mode() == "content":
                st.caption(f"{int(st.session_state['sample_idx'])}/{max_idx}")
            else:
                st.caption(
                    f"{len(task_ids_effective_filtered)} samples in picker (after recomputation + picker filters)"
                )

        current_task_id = task_ids_effective_filtered[
            int(st.session_state["sample_idx"])
        ]

        # Pick a canonical record to source prompt + outputs (first judge with the task).
        canonical_judge = None
        canonical_record: Optional[Dict[str, Any]] = None
        for judge, indexed in index_by_judge.items():
            if current_task_id in indexed:
                canonical_judge = judge
                canonical_record = indexed[current_task_id]
                break
        if canonical_record is None:
            st.error(f"No judge record found for selected task_id: {current_task_id!r}")
            st.stop()

        st.markdown(f"### Sample: `{current_task_id}`")
        st.caption(f"canonical_judge: `{canonical_judge}`")

        prompt_text = _safe_get_record_field(canonical_record, "input_text")
        model_a_name = _safe_get_record_field(
            canonical_record, "model_a_name", default="model_a"
        )
        model_b_name = _safe_get_record_field(
            canonical_record, "model_b_name", default="model_b"
        )
        model_a_output = _safe_get_record_field(canonical_record, "model_a_output")
        model_b_output = _safe_get_record_field(canonical_record, "model_b_output")

        # Judge outputs
        st.markdown("### Judge judgments")
        missing_judges = [
            j
            for j in sorted(index_by_judge.keys())
            if current_task_id not in index_by_judge[j]
        ]
        if missing_judges:
            st.warning(
                "Some judges are missing this sample: "
                + ", ".join(f"`{j}`" for j in missing_judges)
            )

        judges_present = [
            j
            for j in sorted(index_by_judge.keys(), key=str)
            if current_task_id in index_by_judge.get(j, {})
        ]
        _render_compact_judge_summary(
            judges=judges_present,
            index_by_judge=index_by_judge,
            task_id=current_task_id,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            tie_breaker_mode=_dimension_tie_breaker_mode(selected_tie_breaker),
            omit_pairwise_keys=tuple(sorted(str(x) for x in omit_pairwise_keys_stage6)),
            exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
            drop_invalid_samples=bool(drop_invalid_samples),
            dimension_weighted_overall_winner=bool(dimension_weighted_overall_winner),
            dimension_weights_by_user=dimension_weights_by_user,
            correctness_mode=correctness_mode,
            objective_lookup=streamlit_objective_lookup,
            include_plus_correctness=bool(include_plus_correctness),
            use_human_overall_choice=bool(use_human_overall_choice),
        )
        _render_dimension_comparison_table(
            judges=judges_present,
            index_by_judge=index_by_judge,
            task_id=current_task_id,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            tie_breaker_mode=_dimension_tie_breaker_mode(selected_tie_breaker),
            omit_pairwise_keys=tuple(sorted(str(x) for x in omit_pairwise_keys_stage6)),
            exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
        )

        with st.expander("Stage 6 (paper-ready) view for this sample", expanded=False):
            if stage6_backbone_error:
                _emit_pairwise_load_failure_to_ui(
                    exc=stage6_backbone_error,
                    prefix="Stage-6 backbone failed to load",
                    stop_after=False,
                )
            elif stage6_df_selected is None or stage6_df_selected.empty:
                st.warning(
                    "Stage-6 backbone is empty for the current artifact selections and recomputation controls."
                )
            else:
                df_s = stage6_df_selected[
                    stage6_df_selected["_stage6_sample_id"].astype(str)
                    == str(current_task_id)
                ].copy()
                if df_s.empty:
                    st.warning(
                        "This sample_id is not present in the Stage-6-normalized backbone. "
                        "This can happen if Stage 6 normalized task identifiers differently."
                    )
                else:
                    rows: List[Dict[str, Any]] = []
                    for _, rr in df_s.iterrows():
                        row = rr.to_dict()
                        judge_token = str(row.get("_judge_token", "") or "")
                        record_judge_name = str(row.get("judge_model_name", "") or "")
                        winner = stage6_display_winner_name_from_row(row)
                        rows.append(
                            {
                                "judge_token": judge_token,
                                "record_judge_model_name": record_judge_name,
                                "overall_winner": winner,
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows).sort_values("judge_token"),
                        **_width_kwargs(st.dataframe, _ui_width_mode()),
                        hide_index=True,
                    )

                    # Per-dimension winners (after Stage-6 recomputation and omits).
                    dim_cols = [
                        c
                        for c in df_s.columns
                        if str(c).startswith("dim_")
                        and str(c).endswith("_winner_label")
                    ]
                    if not dim_cols:
                        st.caption(
                            "No per-dimension winner columns are available in the Stage-6-normalized rows "
                            "(they may have been omitted)."
                        )
                    else:
                        dims = sorted(
                            {
                                str(c).replace("dim_", "").replace("_winner_label", "")
                                for c in dim_cols
                            }
                        )
                        dim_rows: List[Dict[str, Any]] = []
                        for dim in dims:
                            row_out: Dict[str, Any] = {"dimension": dim}
                            for _, rr in df_s.iterrows():
                                rdict = rr.to_dict()
                                jt = str(rdict.get("_judge_token", "") or "")
                                label = str(
                                    rdict.get(f"dim_{dim}_winner_label", "tie") or "tie"
                                ).strip()
                                if label == "model_a":
                                    disp = str(rdict.get("model_a_name", "model_a"))
                                elif label == "model_b":
                                    disp = str(rdict.get("model_b_name", "model_b"))
                                else:
                                    disp = "tie"
                                row_out[jt] = disp
                            dim_rows.append(row_out)
                        st.dataframe(
                            pd.DataFrame(dim_rows),
                            **_width_kwargs(st.dataframe, _ui_width_mode()),
                            hide_index=True,
                        )

        show_raw = st.checkbox(
            "Show full per-dimension judge details (raw responses)",
            value=False,
            key=f"show_raw_{current_task_id}",
        )
        if show_raw:
            for judge in judges_present:
                record = index_by_judge[judge].get(current_task_id)
                if record is None:
                    continue
                _render_judge_record(
                    judge, record, artifact_by_judge.get(judge, "unknown")
                )

        st.divider()

        _render_text_block(
            "Prompt", prompt_text, height=220, key=f"ta_prompt_{current_task_id}"
        )

        # Model outputs
        out_cols = st.columns(2)
        with out_cols[0]:
            _render_text_block(
                f"{model_a_name} output",
                model_a_output,
                height=380,
                key=f"ta_output_a_{current_task_id}",
            )
        with out_cols[1]:
            _render_text_block(
                f"{model_b_name} output",
                model_b_output,
                height=380,
                key=f"ta_output_b_{current_task_id}",
            )

    with tabs[0]:
        st.markdown("### All samples (overall winners)")
        st.session_state["all_samples_mode"] = _normalize_widget_choice(
            st.session_state.get("all_samples_mode"),
            allowed_values=["single", "aggregate"],
            desired_default="aggregate",
        )
        mode = st.radio(
            "Mode",
            options=["single", "aggregate"],
            horizontal=True,
            key="all_samples_mode",
        )

        if mode == "single":
            judges_all = sorted(index_by_judge.keys(), key=str)
            omit_pairwise_keys, _omit_subjective = normalize_omit_dimensions(
                omit_dimensions
            )
            stage6_df = stage6_df_selected.copy()
            if stage6_backbone_error:
                _emit_pairwise_load_failure_to_ui(
                    exc=stage6_backbone_error,
                    prefix="Stage-6 backbone failed to load",
                    stop_after=False,
                )
                stage6_df = pd.DataFrame()

            # Legacy/explorer outcomes (kept for debugging; not used as primary backbone).
            legacy_task_ids_for_all = list(task_ids)
            outcomes_by_judge_legacy = _build_outcomes_by_judge_for_task_ids(
                judges=judges_all,
                index_by_judge=index_by_judge,
                task_ids=legacy_task_ids_for_all,
                tie_breaker_mode=_dimension_tie_breaker_mode(selected_tie_breaker),
                model_x_name=str(pair.model_a),
                model_y_name=str(pair.model_b),
                omit_pairwise_keys=tuple(
                    sorted(str(x) for x in omit_pairwise_keys_stage6)
                ),
                exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                drop_invalid_samples=bool(drop_invalid_samples),
                dimension_weighted_overall_winner=bool(
                    dimension_weighted_overall_winner
                ),
                dimension_weights_by_user=dimension_weights_by_user,
                correctness_mode=correctness_mode,
                objective_lookup=streamlit_objective_lookup,
                include_plus_correctness=bool(include_plus_correctness),
                use_human_overall_choice=bool(use_human_overall_choice),
            )

            # Validate that this slice is for a single canonical model pair.
            if stage6_df.empty:
                st.warning(
                    "Stage-6 backbone is empty, so Stage-6 tables are unavailable. "
                    "You can still use the legacy/explorer tables below."
                )
                model_x_name = str(pair.model_a)
                model_y_name = str(pair.model_b)
                task_ids_for_all = []
                sample_ids = set()
                outcomes_by_judge = {j: {} for j in judges_all}
                meta_by_sample_id = {}
            else:
                a_vals = sorted(
                    stage6_df["model_a_name"].dropna().astype(str).unique().tolist()
                )
                b_vals = sorted(
                    stage6_df["model_b_name"].dropna().astype(str).unique().tolist()
                )
                if len(a_vals) != 1 or len(b_vals) != 1:
                    st.error(
                        "Stage-6-normalized slice contains multiple model_a/model_b names; "
                        f"model_a={a_vals} model_b={b_vals}. This indicates mixed inputs."
                    )
                    st.stop()

                model_x_name = str(a_vals[0])
                model_y_name = str(b_vals[0])
                st.caption(
                    f"Outcome labels (Stage 6 canonical order): `{model_x_name}`, `{model_y_name}`, plus `tie`."
                )

                try:
                    outcomes_by_judge, meta_by_sample_id = (
                        build_stage6_outcomes_by_judge_token(stage6_df)
                    )
                except Exception as exc:
                    st.error(f"Failed to build Stage-6 backbone outcomes: {exc}")
                    st.stop()

                # Ensure consistent judge-token keys (include empty maps for selected judges).
                for jt in judges_all:
                    outcomes_by_judge.setdefault(str(jt), {})

                sample_ids: set[str] = set()
                for by_sample in outcomes_by_judge.values():
                    sample_ids.update(str(k) for k in by_sample.keys())
                task_ids_for_all = sorted(sample_ids)

            filter_cols = st.columns(5 if _ui_width_mode() == "stretch" else 4)
            with filter_cols[0]:
                task_contains = st.text_input(
                    "task_id contains",
                    value="",
                    key="flt_task_contains",
                )
            with filter_cols[1]:
                judges_selected = st.multiselect(
                    "Judges",
                    options=judges_all,
                    default=judges_all,
                    key="flt_judges",
                )
            with filter_cols[2]:
                judge_mode = st.selectbox(
                    "Judges dropdown behavior",
                    options=["union", "present_only"],
                    index=0,
                    key="flt_judge_mode",
                )
            with filter_cols[3]:
                min_disagree = st.number_input(
                    "min_disagreeing",
                    min_value=0,
                    max_value=max(0, len(judges_all) - 1),
                    value=0,
                    step=1,
                    key="flt_min_disagree",
                )
            if len(filter_cols) > 4:
                with filter_cols[4]:
                    keep_outcomes = st.multiselect(
                        "Keep outcomes",
                        options=[model_x_name, model_y_name, "tie"],
                        default=[model_x_name, model_y_name, "tie"],
                        key="flt_keep_outcomes",
                    )
            else:
                keep_outcomes = [model_x_name, model_y_name, "tie"]

            outcomes_by_judge_filtered = {
                j: outcomes_by_judge.get(j, {}) for j in judges_selected
            }

            filtered_task_ids: List[str] = []
            for tid in task_ids_for_all:
                if task_contains and task_contains not in str(tid):
                    continue
                any_present = False
                any_allowed = False
                for j in judges_selected:
                    val = outcomes_by_judge_filtered.get(j, {}).get(tid)
                    if val is None:
                        continue
                    any_present = True
                    if val in keep_outcomes:
                        any_allowed = True
                if not any_present or not any_allowed:
                    continue
                filtered_task_ids.append(tid)

            item_meta_by_id = {
                tid: {
                    "task_id": str(tid),
                    **(meta_by_sample_id.get(str(tid), {}) or {}),
                }
                for tid in filtered_task_ids
            }
            winners_df = build_overall_winner_table_with_meta(
                item_ids=filtered_task_ids,
                judges=judges_selected,
                outcomes_by_judge=outcomes_by_judge_filtered,
                item_meta_by_id=item_meta_by_id,
            )
            if (
                int(min_disagree) > 0
                and not winners_df.empty
                and "n_disagreeing" in winners_df.columns
            ):
                winners_df = winners_df[
                    winners_df["n_disagreeing"] >= int(min_disagree)
                ].copy()
                filtered_task_ids = (
                    winners_df["item_id"].astype(str).tolist()
                    if "item_id" in winners_df.columns
                    else winners_df["task_id"].astype(str).tolist()
                )

            st.caption(f"Filtered samples: {len(filtered_task_ids)}")

            # Per-judge rates for debugging judge behavior.
            judge_rows: List[Dict[str, Any]] = []
            denom = len(filtered_task_ids)
            for j in judges_selected:
                vals = [
                    outcomes_by_judge_filtered.get(j, {}).get(tid)
                    for tid in filtered_task_ids
                ]
                present = [v for v in vals if v is not None]
                n_present = len(present)
                tie_count = sum(1 for v in present if v == "tie")
                judge_rows.append(
                    {
                        "judge": j,
                        "n_present": int(n_present),
                        "missing_rate": (
                            (1.0 - (n_present / denom)) if denom > 0 else None
                        ),
                        "tie_rate": (tie_count / n_present) if n_present > 0 else None,
                    }
                )
            judge_stats_df = pd.DataFrame(judge_rows)

            if judge_mode == "present_only":
                present_judges = (
                    judge_stats_df[judge_stats_df["n_present"] > 0]["judge"]
                    .astype(str)
                    .tolist()
                )
                outcomes_by_judge_filtered = {
                    j: outcomes_by_judge_filtered.get(j, {}) for j in present_judges
                }
                judges_selected = present_judges

            agreement_df = compute_judge_pair_agreement(
                outcomes_by_judge_filtered,
                task_ids=filtered_task_ids,
                categories=(model_x_name, model_y_name, "tie"),
            )

            st.markdown("### Judge agreement (filtered)")
            st.dataframe(
                agreement_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )
            _render_agreement_mean_std(agreement_df, title="single mode")

            with st.expander("Win rates (pooled + per judge)", expanded=False):
                if st.button("Compute win-rate tables", key="winrate_single_compute"):
                    try:
                        sample_df = _build_pairwise_sample_df_from_named_outcomes(
                            item_ids=filtered_task_ids,
                            outcomes_by_judge=outcomes_by_judge_filtered,
                            user_id=str(selected_persona),
                            variant_label=str(selected_prompt_label),
                            model_a_name=str(model_x_name),
                            model_b_name=str(model_y_name),
                            judge_column="judge",
                        )
                    except Exception as exc:
                        st.error(f"Failed to build win-rate sample frame: {exc}")
                        st.stop()

                    if sample_df.empty:
                        st.warning("No outcomes available to compute win rates.")
                    else:
                        try:
                            matrices = compute_joint_preference_matrices(sample_df)
                            long_frames = [
                                b.pairwise_long
                                for b in matrices.values()
                                if b.pairwise_long is not None
                                and not b.pairwise_long.empty
                            ]
                            pooled_long = (
                                pd.concat(long_frames, ignore_index=True)
                                if long_frames
                                else pd.DataFrame()
                            )
                            by_judge_long = compute_joint_preference_long_by_judge(
                                sample_df, judge_column="judge"
                            )
                        except Exception as exc:
                            st.error(f"Failed to compute win-rate tables: {exc}")
                            st.stop()

                        metric_cols = [
                            "wins",
                            "losses",
                            "ties",
                            "total",
                            "tie_rate",
                            "n_excl_ties",
                            "win_rate",
                            "p_value",
                            "significant",
                            "formatted_win_rate",
                            "formatted_win_rate_personalized_vs_baselines",
                            "objective_flip_rate_vs_original",
                            "objective_reverse_flip_rate_vs_original",
                        ]
                        key_cols = ["persona", "prompt_type", "row_model", "col_model"]

                        st.markdown("### Overall (pooled across judges)")
                        if pooled_long is None or pooled_long.empty:
                            st.warning("Pooled win-rate table is empty.")
                        else:
                            try:
                                pooled_long = _maybe_attach_paired_stats(
                                    sample_df=sample_df,
                                    long_df=pooled_long,
                                    group_by_judge=False,
                                    judge_column="judge",
                                    paired_alpha=0.05,
                                )
                            except Exception as exc:
                                st.error(
                                    "Failed to attach Stage-6-style paired stats to pooled win-rate table: "
                                    f"{exc}"
                                )
                                st.stop()
                            view = pooled_long[
                                pooled_long["row_model"]
                                .astype(str)
                                .isin({str(model_x_name), str(model_y_name)})
                            ].copy()
                            cols = [
                                c for c in (key_cols + metric_cols) if c in view.columns
                            ]
                            st.dataframe(
                                view[cols],
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

                        st.markdown("### Per judge")
                        if by_judge_long is None or by_judge_long.empty:
                            st.warning("Per-judge win-rate table is empty.")
                        else:
                            try:
                                by_judge_long = _maybe_attach_paired_stats(
                                    sample_df=sample_df,
                                    long_df=by_judge_long,
                                    group_by_judge=True,
                                    judge_column="judge",
                                    paired_alpha=0.05,
                                )
                            except Exception as exc:
                                st.error(
                                    "Failed to attach Stage-6-style paired stats to per-judge win-rate table: "
                                    f"{exc}"
                                )
                                st.stop()
                            view = by_judge_long.copy()
                            cols = [
                                c
                                for c in (["judge"] + key_cols + metric_cols)
                                if c in view.columns
                            ]
                            st.dataframe(
                                view[cols],
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

            st.markdown("### Judge summary (filtered)")
            st.dataframe(
                judge_stats_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )

            st.markdown("### Overall winners table (filtered)")
            st.dataframe(
                winners_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )

            with st.expander(
                "Stage 6 paper-ready tables (joint_preference + judge_agreement)",
                expanded=False,
            ):
                jp_view = st.selectbox(
                    "Joint preference view",
                    options=["pooled", "by_judge"],
                    index=0,
                    key="jp_single_view_stage6",
                )
                if st.button("Compute Stage 6 tables", key="jp_single_compute_stage6"):
                    keep = set(str(x) for x in filtered_task_ids)
                    df_all = stage6_df[
                        stage6_df["_stage6_sample_id"].astype(str).isin(keep)
                    ].copy()
                    if df_all.empty:
                        st.warning("No Stage-6 rows remain after filtering.")
                    else:
                        st.markdown(
                            "### Joint preference long table (Stage 6 semantics)"
                        )
                        if jp_view == "pooled":
                            bundles = compute_joint_preference_matrices(df_all)
                            long_frames = [
                                b.pairwise_long
                                for b in bundles.values()
                                if b.pairwise_long is not None
                                and not b.pairwise_long.empty
                            ]
                            long_df = (
                                pd.concat(long_frames, ignore_index=True)
                                if long_frames
                                else pd.DataFrame()
                            )
                        else:
                            long_df = compute_joint_preference_long_by_judge(df_all)

                        if long_df is None or long_df.empty:
                            st.warning(
                                "Joint preference long table is empty for this selection."
                            )
                        else:
                            st.dataframe(
                                long_df,
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

                        st.markdown(
                            "### Judge agreement condition summary (Stage 6 semantics)"
                        )
                        try:
                            artifacts = compute_judge_agreement_for_joint_preference(
                                df_all, n_bootstrap=1000, seed=42
                            )
                            cond = artifacts.get(
                                "agreement_condition_summary", pd.DataFrame()
                            )
                        except Exception as exc:
                            st.error(
                                f"Failed to compute Stage-6 judge agreement: {exc}"
                            )
                            cond = pd.DataFrame()
                        if cond is None or cond.empty:
                            st.warning(
                                "Stage-6 judge agreement condition summary is empty (need >=2 judges with overlap)."
                            )
                        else:
                            # Focus display on this model pair; show all prompt types.
                            focus = cond[
                                (cond["row_model"].astype(str) == str(model_x_name))
                                & (cond["col_model"].astype(str) == str(model_y_name))
                            ].copy()
                            show_df = focus if not focus.empty else cond
                            st.dataframe(
                                show_df,
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

            with st.expander("Explorer (legacy) tables", expanded=False):
                st.caption(
                    "These tables use the legacy Streamlit outcome recomputation over raw "
                    "Stage-5b artifacts (keyed by record.task_id). They may differ from Stage 6."
                )
                legacy_outcomes_filtered = {
                    j: outcomes_by_judge_legacy.get(j, {}) for j in judges_selected
                }
                legacy_filtered = list(legacy_task_ids_for_all)
                if bool(drop_invalid_samples):
                    legacy_filtered = [
                        tid
                        for tid in legacy_filtered
                        if any(
                            str(tid) in legacy_outcomes_filtered.get(j, {})
                            for j in judges_selected
                        )
                    ]
                legacy_agreement = compute_judge_pair_agreement(
                    legacy_outcomes_filtered,
                    task_ids=legacy_filtered,
                    categories=(str(pair.model_a), str(pair.model_b), "tie"),
                )
                st.markdown("### Judge agreement (legacy)")
                st.dataframe(
                    legacy_agreement,
                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                    hide_index=True,
                )
                legacy_winners = build_overall_winner_table(
                    task_ids=legacy_filtered,
                    judges=judges_selected,
                    outcomes_by_judge=legacy_outcomes_filtered,
                )
                st.markdown("### Overall winners table (legacy)")
                st.dataframe(
                    legacy_winners,
                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                    hide_index=True,
                )
        else:
            st.caption(
                "Aggregate across multiple configuration slices and model pairs. "
                "Global agreement uses position-based outcomes (`model_1` vs `model_2` vs `tie`)."
            )
            st.caption(
                "Recomputation controls (omit/exclude/drop) apply to all tie-breaker modes."
            )
            use_human_overall_choice = st.checkbox(
                "Use direct `human_overall_choice` for human judges",
                value=bool(st.session_state.get("agg_use_human_overall_choice", False)),
                key="agg_use_human_overall_choice",
                help=(
                    "For human annotator judges, use `metadata.human_overall_choice` "
                    "instead of the dimension-computed `overall_choice`."
                ),
            )
            if bool(use_human_overall_choice):
                st.caption(
                    "When enabled, human-annotator overall winners come directly from "
                    "`human_overall_choice`; omit-dimension, weighted-win, and "
                    "correctness-based recomputation are not applied to those human "
                    "overall labels."
                )
            agg_weighted = st.checkbox(
                "dimension_weighted_overall_winner (global)",
                value=bool(dimension_weighted_overall_winner),
                key="agg_dimension_weighted_overall_winner",
                help=(
                    "Shortcut toggle for the global sidebar setting. "
                    "When enabled, per-sample winners use persona YAML dimension weights."
                ),
            )
            if bool(agg_weighted) != bool(dimension_weighted_overall_winner):
                st.session_state["dimension_weighted_overall_winner"] = bool(
                    agg_weighted
                )
                st.rerun()

            all_personas = sorted({c.persona for c in configs}, key=str)
            all_gens = sorted({c.generator_model for c in configs}, key=str)
            all_filters = sorted({c.filter_model for c in configs}, key=str)
            all_prompts = sorted(
                {format_prompt_type(c.prompt_type) for c in configs}, key=str
            )
            all_judgment_types = _available_judgment_types(configs)
            aggregate_state_run_dir_key = "agg_filter_selection_run_dir"
            aggregate_run_dir_changed = (
                st.session_state.get(aggregate_state_run_dir_key) != base_dir_str
            )
            st.session_state["agg_personas"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.personas or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_personas")
                ),
                options=all_personas,
                desired_default=all_personas,
            )
            st.session_state["agg_gen_models"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.generator_models or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_gen_models")
                ),
                options=all_gens,
                desired_default=all_gens,
            )
            st.session_state["agg_filter_models"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.filter_models or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_filter_models")
                ),
                options=all_filters,
                desired_default=all_filters,
            )
            st.session_state["agg_prompt_types"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.prompt_types or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_prompt_types")
                ),
                options=all_prompts,
                desired_default=all_prompts,
            )
            st.session_state["agg_judgment_types"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.pairwise_judgment_types or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_judgment_types")
                ),
                options=all_judgment_types,
                desired_default=all_judgment_types,
            )
            st.session_state[aggregate_state_run_dir_key] = base_dir_str

            if _ui_width_mode() == "stretch":
                sel_cols = st.columns(5)
                with sel_cols[0]:
                    sel_personas = st.multiselect(
                        "personas",
                        options=all_personas,
                        key="agg_personas",
                    )
                with sel_cols[1]:
                    sel_gens = st.multiselect(
                        "gen_models",
                        options=all_gens,
                        key="agg_gen_models",
                    )
                with sel_cols[2]:
                    sel_filters = st.multiselect(
                        "filter_models",
                        options=all_filters,
                        key="agg_filter_models",
                    )
                with sel_cols[3]:
                    sel_prompts = st.multiselect(
                        "prompt_types",
                        options=all_prompts,
                        key="agg_prompt_types",
                    )
                with sel_cols[4]:
                    sel_judgment_types = st.multiselect(
                        "pairwise_judgment_type",
                        options=all_judgment_types,
                        key="agg_judgment_types",
                    )
            else:
                sel_row1 = st.columns(2)
                with sel_row1[0]:
                    sel_personas = st.multiselect(
                        "personas",
                        options=all_personas,
                        key="agg_personas",
                    )
                with sel_row1[1]:
                    sel_gens = st.multiselect(
                        "gen_models",
                        options=all_gens,
                        key="agg_gen_models",
                    )
                sel_row2 = st.columns(2)
                with sel_row2[0]:
                    sel_filters = st.multiselect(
                        "filter_models",
                        options=all_filters,
                        key="agg_filter_models",
                    )
                with sel_row2[1]:
                    sel_prompts = st.multiselect(
                        "prompt_types",
                        options=all_prompts,
                        key="agg_prompt_types",
                    )
                sel_row3 = st.columns(1)
                with sel_row3[0]:
                    sel_judgment_types = st.multiselect(
                        "pairwise_judgment_type",
                        options=all_judgment_types,
                        key="agg_judgment_types",
                    )

            selected_configs = _filter_aggregate_pairwise_configs(
                configs=configs,
                personas=sel_personas,
                generator_models=sel_gens,
                filter_models=sel_filters,
                prompt_labels=sel_prompts,
                judgment_types=sel_judgment_types,
            )
            if not selected_configs:
                st.error("No configurations match the selected aggregate filters.")
                st.stop()
            logger.info(
                "resolved aggregate config filters: personas=%s gen_models=%s filter_models=%s prompt_types=%s pairwise_judgment_types=%s selected_config_count=%d",
                sel_personas,
                sel_gens,
                sel_filters,
                sel_prompts,
                sel_judgment_types,
                len(selected_configs),
            )
            st.caption(
                "Selected configurations: "
                f"{len(selected_configs)} "
                f"(pairwise_judgment_type: {', '.join(sel_judgment_types)})"
            )
            if (
                bool(dimension_weighted_overall_winner)
                and dimension_weights_by_user is not None
            ):
                required_user_ids = sorted(
                    {
                        persona_weight_catalog["alias_to_user_id"].get(
                            str(c.persona), str(c.persona)
                        )
                        for c in selected_configs
                    }
                )
                missing_user_ids = [
                    uid
                    for uid in required_user_ids
                    if uid not in dimension_weights_by_user
                ]
                if missing_user_ids:
                    st.error(
                        "Selected persona-weight configs do not cover all personas in "
                        "the current aggregate scope. "
                        f"missing_user_ids={missing_user_ids!r} "
                        f"loaded_keys={sorted(dimension_weights_by_user.keys())!r}"
                    )
                    st.stop()

            # Discover available model pairs across selected configs.
            pair_set: Dict[Tuple[str, str], ModelPairKey] = {}
            for cfg in selected_configs:
                try:
                    pairs_dicts = _cached_discover_pairs(base_dir_str, asdict(cfg))
                except Exception as exc:
                    st.error(f"Failed to discover pairs for a config: {exc}")
                    st.stop()
                for p in pairs_dicts:
                    key = (str(p.get("model_a")), str(p.get("model_b")))
                    pair_set[key] = ModelPairKey(model_a=key[0], model_b=key[1])

            if not pair_set:
                st.error("No model pairs found under the selected configurations.")
                st.stop()

            pair_options = sorted(pair_set.values(), key=lambda x: x.label)
            pair_labels = [p.label for p in pair_options]
            st.session_state["agg_model_pairs"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.model_pairs or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_model_pairs")
                ),
                options=pair_labels,
                desired_default=pair_labels,
            )
            sel_pair_labels = st.multiselect(
                "model_pairs",
                options=pair_labels,
                key="agg_model_pairs",
            )
            selected_pairs = [p for p in pair_options if p.label in sel_pair_labels]
            if not selected_pairs:
                st.error("No model pairs selected.")
                st.stop()

            cfg_dicts = [asdict(c) for c in selected_configs]
            pair_dicts = [asdict(p) for p in selected_pairs]
            judges_union = _cached_discover_judges_union(
                base_dir_str, cfg_dicts, pair_dicts
            )
            if not judges_union:
                st.error("No judges found for the selected aggregate scope.")
                st.stop()

            human_judges_in_scope, llm_judges_in_scope = split_judges_by_group(
                judges_union
            )

            st.session_state["agg_judges"] = _normalize_multiselect_state(
                (
                    saved_aggregate_filters.judges or None
                    if aggregate_run_dir_changed
                    else st.session_state.get("agg_judges")
                ),
                options=judges_union,
                desired_default=judges_union,
            )
            judges_selected = st.multiselect(
                "judges",
                options=judges_union,
                key="agg_judges",
            )
            if not judges_selected:
                st.error("No judges selected.")
                st.stop()

            current_aggregate_selection = AggregateFilterSelection(
                personas=[str(x) for x in sel_personas],
                generator_models=[str(x) for x in sel_gens],
                filter_models=[str(x) for x in sel_filters],
                prompt_types=[str(x) for x in sel_prompts],
                pairwise_judgment_types=[str(x) for x in sel_judgment_types],
                model_pairs=[str(x) for x in sel_pair_labels],
                judges=[str(x) for x in judges_selected],
            )
            if current_aggregate_selection != saved_aggregate_filters:
                try:
                    save_aggregate_filter_selection(
                        aggregate_filter_selection_path,
                        selection=current_aggregate_selection,
                        app_version=None,
                    )
                except Exception as exc:
                    st.error(f"Failed to save aggregate filter selection: {exc}")
                    st.sidebar.caption(
                        f"selection file path: `{aggregate_filter_selection_path}`"
                    )
                    st.stop()

            aggregate_human_annotated_only = st.checkbox(
                "Only samples with human annotations",
                value=False,
                key="agg_only_human_annotated",
                help=(
                    "Restrict Aggregate mode to samples that have at least one "
                    "judgment from a human-annotation judge directory "
                    "(`judge_model_human_annotator*`)."
                ),
            )
            st.caption(
                "Aggregate scope judges: "
                f"{len(human_judges_in_scope)} human, {len(llm_judges_in_scope)} LLM."
            )

            table_shape = st.radio(
                "Table shape",
                options=["wide", "long"],
                index=0,
                horizontal=True,
                key="agg_table_shape",
            )

            judges_to_load = sorted(
                set(str(j) for j in judges_selected)
                | (
                    set(str(j) for j in human_judges_in_scope)
                    if human_judges_in_scope
                    else set()
                )
            )

            try:
                (
                    item_meta_by_id,
                    display_outcomes_by_judge,
                    positional_outcomes_by_judge,
                    human_confidence_by_judge,
                    artifacts_used_rows,
                ) = _cached_load_aggregated_outcomes(
                    base_dir_str=base_dir_str,
                    cfg_dicts=cfg_dicts,
                    pair_dicts=pair_dicts,
                    judges=judges_to_load,
                    tie_breaker_mode=selected_tie_breaker,
                    selection_path_str=str(selection_path),
                    selection_file_mtime=float(
                        selection_path.stat().st_mtime
                        if selection_path.exists()
                        else 0.0
                    ),
                    stage6_tie_breaker_mode=str(stage6_tie_breaker),
                    stage6_omit_pairwise_keys=tuple(
                        sorted(
                            str(x)
                            for x in normalize_omit_dimensions(omit_dimensions)[0]
                        )
                    ),
                    stage6_exclude_eval_parse_errors=bool(exclude_eval_parse_errors),
                    stage6_drop_invalid_samples=bool(drop_invalid_samples),
                    dimension_weighted_overall_winner=bool(
                        dimension_weighted_overall_winner
                    ),
                    dimension_weight_config_paths=tuple(selected_weight_yaml_abs),
                    dimension_weight_selection_mtime=float(
                        persona_weight_selection_mtime
                    ),
                    correctness_mode=correctness_mode,
                    include_plus_correctness=bool(include_plus_correctness),
                    use_human_overall_choice=bool(use_human_overall_choice),
                )
            except Exception as exc:
                _emit_pairwise_load_failure_to_ui(
                    exc=exc,
                    prefix="Failed to load aggregated outcomes",
                    stop_after=True,
                )

            item_ids_all = sorted(item_meta_by_id.keys(), key=str)
            aggregate_failure_rows = [
                row
                for row in artifacts_used_rows
                if str(row.get("load_status", "")).strip().lower() == "failed"
            ]
            if aggregate_failure_rows:
                st.warning(
                    "Some aggregate pairwise artifacts failed to load and were excluded from the table. "
                    f"failed={len(aggregate_failure_rows)} loaded={len(artifacts_used_rows) - len(aggregate_failure_rows)}"
                )
                with st.expander("Aggregate artifact load failures", expanded=False):
                    st.dataframe(
                        pd.DataFrame(aggregate_failure_rows),
                        **_width_kwargs(st.dataframe, _ui_width_mode()),
                        hide_index=True,
                    )
            with st.expander("Artifacts used (aggregate)", expanded=False):
                if artifacts_used_rows:
                    st.dataframe(
                        pd.DataFrame(artifacts_used_rows),
                        **_width_kwargs(st.dataframe, _ui_width_mode()),
                        hide_index=True,
                    )
                else:
                    st.caption("No artifacts were recorded as used (unexpected).")

            filter_cols = st.columns(5 if _ui_width_mode() == "stretch" else 4)
            with filter_cols[0]:
                task_contains = st.text_input(
                    "task_id contains",
                    value="",
                    key="agg_flt_task_contains",
                )
            with filter_cols[1]:
                model_pair_contains = st.text_input(
                    "model_pair contains",
                    value="",
                    key="agg_flt_pair_contains",
                )
            with filter_cols[2]:
                judge_mode = st.selectbox(
                    "Judges dropdown behavior",
                    options=["union", "present_only"],
                    index=0,
                    key="agg_flt_judge_mode",
                )
            with filter_cols[3]:
                min_disagree = st.number_input(
                    "min_disagreeing",
                    min_value=0,
                    max_value=max(0, len(judges_selected) - 1),
                    value=0,
                    step=1,
                    key="agg_flt_min_disagree",
                )
            if len(filter_cols) > 4:
                with filter_cols[4]:
                    keep_positional = st.multiselect(
                        "Keep positional outcomes",
                        options=["model_1", "model_2", "tie"],
                        default=["model_1", "model_2", "tie"],
                        key="agg_flt_keep_positional",
                    )
            else:
                keep_positional = ["model_1", "model_2", "tie"]

            judges_selected_sorted = sorted(judges_selected, key=str)
            display_outcomes_filtered = {
                j: display_outcomes_by_judge.get(j, {}) for j in judges_selected_sorted
            }
            positional_outcomes_filtered = {
                j: positional_outcomes_by_judge.get(j, {})
                for j in judges_selected_sorted
            }

            filtered_item_ids: List[str] = []
            for item_id in item_ids_all:
                meta = item_meta_by_id.get(item_id, {})
                tid = str(meta.get("task_id", ""))
                mp = str(meta.get("model_pair", ""))
                if task_contains and task_contains not in tid:
                    continue
                if model_pair_contains and model_pair_contains not in mp:
                    continue

                any_present = False
                any_allowed = False
                for j in judges_selected_sorted:
                    val = positional_outcomes_filtered.get(j, {}).get(item_id)
                    if val is None:
                        continue
                    any_present = True
                    if val in keep_positional:
                        any_allowed = True
                if not any_present or not any_allowed:
                    continue
                filtered_item_ids.append(item_id)

            human_judges_loaded = [
                j
                for j in sorted(display_outcomes_by_judge.keys(), key=str)
                if is_human_judge_token(j)
            ]
            available_human_confidence_levels = sorted(
                {
                    str(confidence).strip()
                    for judge in human_judges_loaded
                    for confidence in human_confidence_by_judge.get(judge, {}).values()
                    if str(confidence).strip()
                },
                key=str,
            )
            human_confidence_levels_selected = st.multiselect(
                "human_overall_confidence",
                options=available_human_confidence_levels,
                default=available_human_confidence_levels,
                key="agg_human_overall_confidence",
                help=(
                    "Restrict Aggregate mode to items that have at least one human "
                    "annotator judgment whose `human_overall_confidence` matches one "
                    "of the selected levels."
                ),
                disabled=not bool(available_human_confidence_levels),
            )
            if not available_human_confidence_levels:
                st.caption(
                    "No human_overall_confidence values are available in the current aggregate scope."
                )
            if aggregate_human_annotated_only:
                if not human_judges_loaded:
                    st.warning(
                        "Human-annotated-only filter is enabled, but no human "
                        "judge directories were found in the selected aggregate scope."
                    )
                    filtered_item_ids = []
                else:
                    filtered_item_ids = filter_item_ids_to_human_annotated(
                        filtered_item_ids,
                        outcomes_by_judge=display_outcomes_by_judge,
                        human_judges=human_judges_loaded,
                    )
            if (
                human_judges_loaded
                and available_human_confidence_levels
                and set(human_confidence_levels_selected)
                != set(available_human_confidence_levels)
            ):
                filtered_item_ids = filter_item_ids_by_human_confidence(
                    filtered_item_ids,
                    human_confidence_by_judge=human_confidence_by_judge,
                    human_judges=human_judges_loaded,
                    allowed_confidence_levels=human_confidence_levels_selected,
                )

            def _judge_stats(judges_list: List[str], ids: List[str]) -> pd.DataFrame:
                denom = len(ids)
                rows: List[Dict[str, Any]] = []
                for j in judges_list:
                    vals = [
                        positional_outcomes_filtered.get(j, {}).get(item_id)
                        for item_id in ids
                    ]
                    present = [v for v in vals if v is not None]
                    n_present = len(present)
                    tie_count = sum(1 for v in present if v == "tie")
                    rows.append(
                        {
                            "judge": j,
                            "n_present": int(n_present),
                            "missing_rate": (
                                (1.0 - (n_present / denom)) if denom > 0 else None
                            ),
                            "tie_rate": (
                                (tie_count / n_present) if n_present > 0 else None
                            ),
                        }
                    )
                return pd.DataFrame(rows)

            judge_stats_df = _judge_stats(judges_selected_sorted, filtered_item_ids)

            judges_effective = list(judges_selected_sorted)
            if judge_mode == "present_only" and not judge_stats_df.empty:
                judges_effective = (
                    judge_stats_df[judge_stats_df["n_present"] > 0]["judge"]
                    .astype(str)
                    .tolist()
                )

            display_outcomes_filtered = {
                j: display_outcomes_by_judge.get(j, {}) for j in judges_effective
            }
            positional_outcomes_filtered = {
                j: positional_outcomes_by_judge.get(j, {}) for j in judges_effective
            }

            # Separate LLM judges from human judges for win-rate tables.
            # Human judges are excluded from winner/comparison tables but
            # remain in judges_effective for agreement analyses.
            _human_eff, llm_judges_for_tables = split_judges_by_group(
                judges_effective
            )
            if _human_eff:
                st.caption(
                    f"{len(_human_eff)} human judge(s) excluded from "
                    "win-rate tables (available in agreement analysis)."
                )
            if not llm_judges_for_tables:
                st.warning(
                    "All selected judges are human annotators. "
                    "Win-rate tables require at least one LLM judge."
                )
            llm_display_outcomes = {
                j: display_outcomes_by_judge.get(j, {})
                for j in llm_judges_for_tables
            }

            # Build wide first for derived filters (min_disagreeing).
            winners_wide_df = build_overall_winner_table_with_meta(
                item_ids=filtered_item_ids,
                judges=llm_judges_for_tables,
                outcomes_by_judge=llm_display_outcomes,
                item_meta_by_id=item_meta_by_id,
            )
            if (
                int(min_disagree) > 0
                and not winners_wide_df.empty
                and "n_disagreeing" in winners_wide_df.columns
            ):
                winners_wide_df = winners_wide_df[
                    winners_wide_df["n_disagreeing"] >= int(min_disagree)
                ].copy()
                filtered_item_ids = winners_wide_df["item_id"].astype(str).tolist()

            judge_stats_df = _judge_stats(judges_effective, filtered_item_ids)

            st.caption(f"Filtered items: {len(filtered_item_ids)}")

            global_agreement_df = compute_judge_pair_agreement(
                {j: positional_outcomes_filtered.get(j, {}) for j in judges_effective},
                task_ids=filtered_item_ids,
                categories=("model_1", "model_2", "tie"),
            )

            group_to_item_ids: Dict[str, List[str]] = {}
            for item_id in filtered_item_ids:
                meta = item_meta_by_id.get(item_id, {})
                group = str(meta.get("model_pair", "unknown_pair"))
                group_to_item_ids.setdefault(group, []).append(item_id)
            per_pair_agreement_df = compute_judge_pair_agreement_by_group(
                {j: positional_outcomes_filtered.get(j, {}) for j in judges_effective},
                group_to_item_ids=group_to_item_ids,
                categories=("model_1", "model_2", "tie"),
            )

            st.markdown("### Judge agreement (filtered, global positional)")
            st.dataframe(
                global_agreement_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )
            _render_agreement_mean_std(
                global_agreement_df, title="aggregate global positional"
            )

            human_judges_effective, llm_judges_effective = split_judges_by_group(
                judges_effective
            )
            human_group_agreement_df = filter_agreement_df_to_within_judge_group(
                global_agreement_df,
                judges_in_group=human_judges_effective,
            )
            llm_group_agreement_df = filter_agreement_df_to_within_judge_group(
                global_agreement_df,
                judges_in_group=llm_judges_effective,
            )
            _render_agreement_mean_std(
                llm_group_agreement_df,
                title="aggregate global positional, llm-llm",
            )
            _render_agreement_mean_std(
                human_group_agreement_df,
                title="aggregate global positional, human-human",
            )
            human_llm_agreement_df = filter_agreement_df_between_judge_groups(
                global_agreement_df,
                judges_in_group_a=human_judges_effective,
                judges_in_group_b=llm_judges_effective,
            )
            _render_agreement_mean_std(
                human_llm_agreement_df,
                title="aggregate global positional, human-llm",
            )
            _render_per_human_llm_agreement_summary(
                human_llm_agreement_df,
                human_judges=human_judges_effective,
                llm_judges=llm_judges_effective,
                title="aggregate global positional, per-human vs llm",
            )

            with st.expander("Per-dimension pooled analysis", expanded=False):
                if st.button(
                    "Compute per-dimension agreement tables",
                    key="agg_dim_agreement_compute",
                ):
                    try:
                        omit_keys, _unknown = normalize_omit_dimensions(omit_dimensions)
                        dimension_sample_df = (
                            _build_aggregate_dimension_sample_frame_for_model_pair(
                                selected_model_pair=None,
                                item_ids_for_pair=list(filtered_item_ids),
                                item_meta_by_id=item_meta_by_id,
                                judges_effective=judges_effective,
                                artifacts_used_rows=artifacts_used_rows,
                                tie_breaker_mode=str(selected_tie_breaker),
                                stage6_tie_breaker_mode=str(stage6_tie_breaker),
                                omit_pairwise_keys=tuple(
                                    sorted(str(x) for x in omit_keys)
                                ),
                                exclude_eval_parse_errors=bool(
                                    exclude_eval_parse_errors
                                ),
                                drop_invalid_samples=bool(drop_invalid_samples),
                                dimension_weighted_overall_winner=bool(
                                    dimension_weighted_overall_winner
                                ),
                                dimension_weight_config_paths=tuple(
                                    selected_weight_yaml_abs
                                ),
                                dimension_weight_selection_mtime=float(
                                    persona_weight_selection_mtime
                                ),
                                correctness_mode=correctness_mode,
                                base_dir_str=base_dir_str,
                                include_plus_correctness=bool(
                                    include_plus_correctness
                                ),
                                streamlit_objective_lookup=streamlit_objective_lookup,
                                use_human_overall_choice=bool(
                                    use_human_overall_choice
                                ),
                            )
                        )
                    except Exception as exc:
                        st.error(
                            f"Failed to build pooled per-dimension sample frame: {exc}"
                        )
                        st.stop()

                    if dimension_sample_df is None or dimension_sample_df.empty:
                        st.warning(
                            "No pooled per-dimension rows available under current filters."
                        )
                    else:
                        try:
                            dimension_agreement_df = compute_dimension_judge_pair_agreement(
                                dimension_sample_df
                            )
                        except Exception as exc:
                            st.error(
                                f"Failed to compute per-dimension judge agreement: {exc}"
                            )
                            st.stop()

                        if dimension_agreement_df.empty:
                            st.warning(
                                "No per-dimension judge agreement rows could be computed."
                            )
                        else:
                            st.dataframe(
                                _sort_dimension_table(
                                    dimension_agreement_df,
                                    omit_pairwise_keys=omit_keys,
                                ),
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

                            llm_dimension_agreement_df = (
                                filter_agreement_df_to_within_judge_group(
                                    dimension_agreement_df,
                                    judges_in_group=llm_judges_effective,
                                )
                            )
                            human_dimension_agreement_df = (
                                filter_agreement_df_to_within_judge_group(
                                    dimension_agreement_df,
                                    judges_in_group=human_judges_effective,
                                )
                            )
                            human_llm_dimension_agreement_df = (
                                filter_agreement_df_between_judge_groups(
                                    dimension_agreement_df,
                                    judges_in_group_a=human_judges_effective,
                                    judges_in_group_b=llm_judges_effective,
                                )
                            )

                            _render_dimension_agreement_summary(
                                dimension_agreement_df,
                                title="aggregate global positional",
                                omit_pairwise_keys=omit_keys,
                            )
                            _render_dimension_agreement_summary(
                                llm_dimension_agreement_df,
                                title="aggregate global positional, llm-llm",
                                omit_pairwise_keys=omit_keys,
                            )
                            _render_dimension_agreement_summary(
                                human_dimension_agreement_df,
                                title="aggregate global positional, human-human",
                                omit_pairwise_keys=omit_keys,
                            )
                            _render_dimension_agreement_summary(
                                human_llm_dimension_agreement_df,
                                title="aggregate global positional, human-llm",
                                omit_pairwise_keys=omit_keys,
                            )
                            _render_dimension_per_human_llm_agreement_summary(
                                dimension_agreement_df,
                                human_judges=human_judges_effective,
                                llm_judges=llm_judges_effective,
                                title="aggregate global positional, per-human vs llm",
                                omit_pairwise_keys=omit_keys,
                            )

            with st.expander(
                "Win/tie rates per dimension (selected model pair)", expanded=False
            ):
                pair_keys = sorted(group_to_item_ids.keys(), key=str)
                if not pair_keys:
                    st.caption("No model pairs available under current filters.")
                else:
                    sel_dim_pair = st.selectbox(
                        "model_pair",
                        options=pair_keys,
                        index=0,
                        key="agg_dim_rates_model_pair",
                    )
                    if st.button(
                        "Compute per-dimension rates",
                        key="agg_dim_rates_compute",
                    ):
                        try:
                            omit_keys, _unknown = normalize_omit_dimensions(
                                omit_dimensions
                            )
                            sample_df = _build_aggregate_dimension_sample_frame_for_model_pair(
                                selected_model_pair=str(sel_dim_pair),
                                item_ids_for_pair=group_to_item_ids.get(
                                    str(sel_dim_pair), []
                                ),
                                item_meta_by_id=item_meta_by_id,
                                judges_effective=judges_effective,
                                artifacts_used_rows=artifacts_used_rows,
                                tie_breaker_mode=str(selected_tie_breaker),
                                stage6_tie_breaker_mode=str(stage6_tie_breaker),
                                omit_pairwise_keys=tuple(
                                    sorted(str(x) for x in omit_keys)
                                ),
                                exclude_eval_parse_errors=bool(
                                    exclude_eval_parse_errors
                                ),
                                drop_invalid_samples=bool(drop_invalid_samples),
                                dimension_weighted_overall_winner=bool(
                                    dimension_weighted_overall_winner
                                ),
                                dimension_weight_config_paths=tuple(
                                    selected_weight_yaml_abs
                                ),
                                dimension_weight_selection_mtime=float(
                                    persona_weight_selection_mtime
                                ),
                                correctness_mode=correctness_mode,
                                base_dir_str=base_dir_str,
                                include_plus_correctness=bool(include_plus_correctness),
                                streamlit_objective_lookup=streamlit_objective_lookup,
                                use_human_overall_choice=bool(use_human_overall_choice),
                            )
                        except Exception as exc:
                            st.error(
                                f"Failed to build per-dimension sample frame: {exc}"
                            )
                            st.stop()

                        if sample_df is None or sample_df.empty:
                            st.warning(
                                "No per-dimension rows available under current filters."
                            )
                        else:
                            try:
                                dim_rates = compute_dimension_win_rates(
                                    sample_df,
                                    group_by_variant=False,
                                    group_by_judge=False,
                                    group_by_user=False,
                                )
                            except Exception as exc:
                                st.error(
                                    f"Failed to compute per-dimension win rates: {exc}"
                                )
                                st.stop()

                            if dim_rates.empty:
                                st.warning("No per-dimension rates could be computed.")
                            else:
                                omit_set = {str(x) for x in omit_keys}
                                canonical_order = [
                                    str(d)
                                    for d in PAIRWISE_DIMENSIONS
                                    if str(d) not in omit_set
                                ]
                                # Derive the actual model_1 name for the selected model_pair
                                # so we can label the column meaningfully.
                                model_1_name = None
                                model_1_candidates: set[str] = set()
                                for item_id in group_to_item_ids.get(
                                    str(sel_dim_pair), []
                                ):
                                    meta = item_meta_by_id.get(item_id, {})
                                    if not isinstance(meta, dict):
                                        continue
                                    if str(meta.get("model_pair")) != str(sel_dim_pair):
                                        continue
                                    m1 = str(meta.get("model_1", "")).strip()
                                    if m1:
                                        model_1_candidates.add(m1)
                                if len(model_1_candidates) == 1:
                                    model_1_name = next(iter(model_1_candidates))
                                elif len(model_1_candidates) > 1:
                                    raise ValueError(
                                        "Cannot determine a unique model_1 name for this model_pair under current filters. "
                                        f"model_pair={sel_dim_pair!r} candidates={sorted(model_1_candidates)!r}"
                                    )
                                out = dim_rates[
                                    dim_rates["model_pair"].astype(str)
                                    == str(sel_dim_pair)
                                ].copy()
                                out = out[
                                    [
                                        "dimension",
                                        "total_comparisons",
                                        "model_a_win_rate",
                                        "tie_rate",
                                    ]
                                ].rename(
                                    columns={
                                        "model_a_win_rate": (
                                            f"{model_1_name}_win_rate"
                                            if model_1_name
                                            else "model_1_win_rate"
                                        )
                                    }
                                )
                                if not out.empty:
                                    out["dimension"] = pd.Categorical(
                                        out["dimension"].astype(str),
                                        categories=canonical_order,
                                        ordered=True,
                                    )
                                    out = out.sort_values(
                                        by=["dimension"], ascending=True
                                    )
                                st.dataframe(
                                    out,
                                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                                    hide_index=True,
                                )

            with st.expander("Win rates (pooled + per judge)", expanded=False):
                if st.button("Compute win-rate tables", key="winrate_agg_compute"):
                    try:
                        sample_df = _build_pairwise_sample_df_from_positional_outcomes(
                            item_ids=filtered_item_ids,
                            outcomes_by_judge={
                                j: positional_outcomes_filtered.get(j, {})
                                for j in judges_effective
                            },
                            item_meta_by_id=item_meta_by_id,
                            judge_column="judge",
                        )
                    except Exception as exc:
                        st.error(f"Failed to build win-rate sample frame: {exc}")
                        st.stop()

                    if sample_df.empty:
                        st.warning("No outcomes available to compute win rates.")
                    else:
                        try:
                            matrices = compute_joint_preference_matrices(sample_df)
                            long_frames = [
                                b.pairwise_long
                                for b in matrices.values()
                                if b.pairwise_long is not None
                                and not b.pairwise_long.empty
                            ]
                            pooled_long = (
                                pd.concat(long_frames, ignore_index=True)
                                if long_frames
                                else pd.DataFrame()
                            )
                            by_judge_long = compute_joint_preference_long_by_judge(
                                sample_df, judge_column="judge"
                            )
                        except Exception as exc:
                            st.error(f"Failed to compute win-rate tables: {exc}")
                            st.stop()

                        metric_cols = [
                            "wins",
                            "losses",
                            "ties",
                            "total",
                            "tie_rate",
                            "n_excl_ties",
                            "win_rate",
                            "p_value",
                            "significant",
                            "formatted_win_rate",
                            "formatted_win_rate_personalized_vs_baselines",
                            "objective_flip_rate_vs_original",
                            "objective_reverse_flip_rate_vs_original",
                        ]
                        key_cols = ["persona", "prompt_type", "row_model", "col_model"]

                        st.markdown("### Overall (pooled across judges)")
                        if pooled_long is None or pooled_long.empty:
                            st.warning("Pooled win-rate table is empty.")
                        else:
                            try:
                                pooled_long = _maybe_attach_paired_stats(
                                    sample_df=sample_df,
                                    long_df=pooled_long,
                                    group_by_judge=False,
                                    judge_column="judge",
                                    paired_alpha=0.05,
                                )
                            except Exception as exc:
                                st.error(
                                    "Failed to attach Stage-6-style paired stats to pooled win-rate table: "
                                    f"{exc}"
                                )
                                st.stop()
                            cols = [
                                c
                                for c in (key_cols + metric_cols)
                                if c in pooled_long.columns
                            ]
                            st.dataframe(
                                pooled_long[cols],
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

                        st.markdown("### Per judge")
                        if by_judge_long is None or by_judge_long.empty:
                            st.warning("Per-judge win-rate table is empty.")
                        else:
                            try:
                                by_judge_long = _maybe_attach_paired_stats(
                                    sample_df=sample_df,
                                    long_df=by_judge_long,
                                    group_by_judge=True,
                                    judge_column="judge",
                                    paired_alpha=0.05,
                                )
                            except Exception as exc:
                                st.error(
                                    "Failed to attach Stage-6-style paired stats to per-judge win-rate table: "
                                    f"{exc}"
                                )
                                st.stop()
                            cols = [
                                c
                                for c in (["judge"] + key_cols + metric_cols)
                                if c in by_judge_long.columns
                            ]
                            st.dataframe(
                                by_judge_long[cols],
                                **_width_kwargs(st.dataframe, _ui_width_mode()),
                                hide_index=True,
                            )

            st.markdown("### Judge agreement (filtered, per model pair positional)")
            st.dataframe(
                per_pair_agreement_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )

            st.markdown("### Judge summary (filtered)")
            st.dataframe(
                judge_stats_df,
                **_width_kwargs(st.dataframe, _ui_width_mode()),
                hide_index=True,
            )

            st.markdown("### Overall winners table (filtered)")
            if table_shape == "wide":
                display_wide_df = winners_wide_df.copy()
                # When human-annotated-only filter is active and there are
                # effective human judges, append a per-human-judge outcome
                # column and a n_human_disagreeing summary column so the
                # table shows LLM and human votes side-by-side.
                if aggregate_human_annotated_only and _human_eff:
                    human_display_outcomes_eff = {
                        j: display_outcomes_by_judge.get(j, {})
                        for j in _human_eff
                    }
                    n_human_disagreeing_vals: List[int] = []
                    for iid in display_wide_df["item_id"].astype(str):
                        human_present = [
                            v
                            for j in _human_eff
                            for v in [human_display_outcomes_eff.get(j, {}).get(iid)]
                            if v is not None
                        ]
                        unique_human = sorted({x for x in human_present})
                        n_human_disagreeing_vals.append(
                            int(max(0, len(unique_human) - 1))
                        )
                    display_wide_df["n_human_disagreeing"] = n_human_disagreeing_vals
                    for j in _human_eff:
                        j_outcomes = human_display_outcomes_eff.get(j, {})
                        display_wide_df[str(j)] = (
                            display_wide_df["item_id"].astype(str).map(j_outcomes)
                        )
                st.dataframe(
                    display_wide_df,
                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                    hide_index=True,
                )
            else:
                winners_long_df = build_long_winner_table_with_meta(
                    item_ids=filtered_item_ids,
                    judges=llm_judges_for_tables,
                    outcomes_by_judge=llm_display_outcomes,
                    item_meta_by_id=item_meta_by_id,
                )
                st.dataframe(
                    winners_long_df,
                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                    hide_index=True,
                )

            if _is_stage6_tie_breaker(selected_tie_breaker):
                with st.expander(
                    "Stage 6 joint_preference_long-like table (for CSV comparison)",
                    expanded=False,
                ):
                    jp_view = st.selectbox(
                        "View",
                        options=["pooled", "by_judge"],
                        index=0,
                        key="jp_agg_view",
                    )
                    if st.button("Compute", key="jp_agg_compute"):
                        keep = set(
                            str(item_meta_by_id.get(item_id, {}).get("task_id", ""))
                            for item_id in filtered_item_ids
                        )
                        omit_pairwise_keys, _omit_subjective = (
                            normalize_omit_dimensions(omit_dimensions)
                        )
                        stage6_tie = str(stage6_tie_breaker)
                        rows_all: List[Dict[str, Any]] = []

                        for cfg in selected_configs:
                            for p in selected_pairs:
                                for judge in judges_effective:
                                    artifacts = (
                                        list_pairwise_artifacts_for_pair_and_judge(
                                            Path(base_dir_str).expanduser().resolve(),
                                            cfg,
                                            p,
                                            judge,
                                        )
                                    )
                                    if not artifacts:
                                        continue
                                    stage6_default = choose_stage6_default_artifact(
                                        artifacts
                                    )
                                    stage6_default_str = str(stage6_default)

                                    key = PairwiseArtifactSelectionKey(
                                        persona=str(cfg.persona),
                                        generator_model=str(cfg.generator_model),
                                        filter_model=str(cfg.filter_model),
                                        prompt_type=str(
                                            format_prompt_type(cfg.prompt_type)
                                        ),
                                        pairwise_judgment_type=str(
                                            cfg.pairwise_judgment_type
                                        ),
                                        model_a=str(p.model_a),
                                        model_b=str(p.model_b),
                                        judge_model=str(judge),
                                    )
                                    saved_selected = get_saved_selected_path(
                                        selections_by_key, key
                                    )
                                    selected_path_str = stage6_default_str
                                    if saved_selected is not None:
                                        saved_resolved = str(
                                            Path(saved_selected).expanduser().resolve()
                                        )
                                        candidates = {str(pp) for pp in artifacts}
                                        if saved_resolved not in candidates:
                                            raise ValueError(
                                                "Saved artifact selection does not match current candidates. "
                                                f"key={key.to_string()!r} saved={saved_resolved!r}"
                                            )
                                        selected_path_str = saved_resolved

                                    rows = _cached_load_stage6_pairwise_rows(
                                        str(selected_path_str),
                                        tie_breaker_mode=stage6_tie,
                                        omit_pairwise_keys=tuple(
                                            sorted(str(x) for x in omit_pairwise_keys)
                                        ),
                                        exclude_eval_parse_errors=bool(
                                            exclude_eval_parse_errors
                                        ),
                                        drop_invalid_samples=bool(drop_invalid_samples),
                                        judge_token=str(judge),
                                        dimension_weighted_winner=bool(
                                            dimension_weighted_overall_winner
                                        ),
                                        dimension_weight_config_paths=tuple(
                                            selected_weight_yaml_abs
                                        ),
                                        dimension_weight_selection_mtime=float(
                                            persona_weight_selection_mtime
                                        ),
                                        correctness_mode=correctness_mode,
                                        base_dir_for_objective=base_dir_str,
                                        include_plus_correctness=bool(
                                            include_plus_correctness
                                        ),
                                        use_human_overall_choice=bool(
                                            use_human_overall_choice
                                        ),
                                    )
                                    for r in rows:
                                        if (
                                            str(
                                                r.get("_stage6_sample_id")
                                                or stage6_sample_id_from_row(r)
                                            )
                                            in keep
                                        ):
                                            rows_all.append(r)

                        if not rows_all:
                            st.warning(
                                "No Stage-6-normalized rows available for this filtered set."
                            )
                        else:
                            df_all = pd.DataFrame(rows_all)
                            if jp_view == "pooled":
                                bundles = compute_joint_preference_matrices(df_all)
                                long_frames = [
                                    b.pairwise_long
                                    for b in bundles.values()
                                    if b.pairwise_long is not None
                                    and not b.pairwise_long.empty
                                ]
                                long_df = (
                                    pd.concat(long_frames, ignore_index=True)
                                    if long_frames
                                    else pd.DataFrame()
                                )
                            else:
                                long_frames = []
                                for judge_name in sorted(
                                    df_all["judge_model_name"]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist(),
                                    key=str,
                                ):
                                    df_j = df_all[
                                        df_all["judge_model_name"].astype(str)
                                        == str(judge_name)
                                    ].copy()
                                    bundles = compute_joint_preference_matrices(df_j)
                                    for b in bundles.values():
                                        if (
                                            b.pairwise_long is None
                                            or b.pairwise_long.empty
                                        ):
                                            continue
                                        long_frames.append(
                                            b.pairwise_long.assign(
                                                judge_model_name=str(judge_name)
                                            )
                                        )
                                long_df = (
                                    pd.concat(long_frames, ignore_index=True)
                                    if long_frames
                                    else pd.DataFrame()
                                )

                            if long_df.empty:
                                st.warning(
                                    "Joint preference table is empty for this selection."
                                )
                            else:
                                st.dataframe(
                                    long_df,
                                    **_width_kwargs(st.dataframe, _ui_width_mode()),
                                    hide_index=True,
                                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
