"""Logic for personalizing tasks to create a vibe-testing dataset."""

import itertools
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

import tqdm

from .data_utils import (
    UserProfile,
    BenchmarkSample,
    VibeTask,
    ChangeIdentificationOutput,
    ChangeOption,
    PersonalizedSample,
    PromptVariation,
    VerificationOutput,
    FieldChanges,
)
from .models.base import BaseModel, GenerationRequest
from .utils import parse_and_validate_json, load_json, format_test_cases_for_prompt

logger = logging.getLogger(__name__)


class TaskPersonalizer:
    """
    Adapts benchmark samples into personalized VibeTasks based on a user profile
    by using an LLM to generate and verify prompt variations.
    """

    _IDENTIFY_CHANGE_TARGET_FIELDS: List[str] = [
        "input_dimensions.real_world_context_embedding",
        "input_dimensions.persona_based_task_framing",
        "output_dimensions.clarity_and_comprehensibility",
        "output_dimensions.efficiency",
        "output_dimensions.expressive_style",
        "persona.description",
        "context.real_world_context_embedding",
        "context.persona_based_task_framing",
        "preferred_output_dimensions",
    ]

    def __init__(
        self,
        model: BaseModel,
        generation_kwargs: Dict[str, Any] = None,
        batch_size: int = 1,
        artifacts_dir: Optional[str] = None,
    ):
        """
        Initializes the TaskPersonalizer.

        Args:
            model: An instance of a language model for generation tasks.
            generation_kwargs: Default generation parameters for the model.
            artifacts_dir: Optional directory for saving metadata artifacts produced during
                personalization (e.g., identified change options). This directory is
                intentionally separate from dataset sample JSONs to avoid collisions with
                downstream globbing logic.
        """
        self._model = model
        self._generation_kwargs = generation_kwargs or {}
        self._batch_size = max(1, int(batch_size))
        self._artifacts_dir = (
            Path(artifacts_dir).expanduser() if artifacts_dir else None
        )

        # Load prompts from the model's configuration
        prompt_config = self._model.config.get("personalization_prompts", {})
        self._identify_changes_template = prompt_config.get("identify_changes")
        self._compose_prompt_template = prompt_config.get("compose_prompt")
        self._compose_prompt_template_humaneval_plus = prompt_config.get(
            "compose_prompt_humaneval_plus"
        )
        self._verify_prompt_template = prompt_config.get("verify_prompt")

        if not all(
            [
                self._identify_changes_template,
                self._compose_prompt_template,
                self._verify_prompt_template,
            ]
        ):
            raise ValueError(
                "Personalization prompts not fully configured in model config."
            )

        # Load simple personalization prompts (optional; required only for
        # build_simple_dataset).
        simple_config = self._model.config.get("simple_personalization_prompts", {})
        self._simple_compose_template = simple_config.get("compose_prompt")
        self._simple_compose_template_humaneval_plus = simple_config.get(
            "compose_prompt_humaneval_plus"
        )
        self._simple_verify_template = simple_config.get("verify_prompt")

        # Load schemas
        schema_dir = os.path.join(os.path.dirname(__file__), "schemas")
        self._identify_changes_schema = load_json(
            os.path.join(schema_dir, "identify_changes_schema.json")
        )
        self._compose_prompt_schema = load_json(
            os.path.join(schema_dir, "compose_prompt_schema.json")
        )
        self._verify_prompt_schema = load_json(
            os.path.join(schema_dir, "verify_prompt_schema.json")
        )

        logger.info(f"Initialized TaskPersonalizer with model: {model}")

    @staticmethod
    def _sanitize_artifact_token(value: str) -> str:
        """
        Sanitize user-provided tokens for safe filenames.

        Args:
            value: Raw token (e.g., user_id).

        Returns:
            A filesystem-safe token.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-")
        return cleaned or "unknown"

    @staticmethod
    def _next_versioned_path(parent_dir: Path, base_name: str, ext: str) -> Path:
        """
        Resolve a next available versioned filename inside a directory.

        Args:
            parent_dir: Directory to place the artifact.
            base_name: Base filename without version/ext.
            ext: File extension without leading dot.

        Returns:
            Path: A versioned path like ``{base_name}_v00.{ext}``.

        Raises:
            RuntimeError: If no free version slot is available.
        """
        for version in range(100):
            candidate = parent_dir / f"{base_name}_v{version:02d}.{ext}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(
            f"Unable to allocate a versioned artifact filename for '{base_name}' in {parent_dir}."
        )

    def _persist_identified_changes(
        self,
        *,
        profile: UserProfile,
        parsed_json: Dict[str, Any],
        raw_response: str,
    ) -> None:
        """
        Persist the identified changes output for later reference.

        This intentionally writes into a dedicated metadata directory so Stage 4 / experiment
        scanners that glob ``dataset_dir/*.json`` do not treat this file as a dataset sample.

        Args:
            profile: User profile used for identifying changes.
            parsed_json: Parsed JSON payload (validated against schema).
            raw_response: Raw model output string used to produce ``parsed_json``.

        Raises:
            RuntimeError: If the artifact cannot be written.
        """
        if self._artifacts_dir is None:
            return

        try:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to create personalization artifacts directory: {self._artifacts_dir}"
            ) from exc

        safe_user_id = self._sanitize_artifact_token(profile.user_id or "unknown_user")
        base_name = f"identify_changes_user-{safe_user_id}"
        output_path = self._next_versioned_path(self._artifacts_dir, base_name, "json")

        payload = {
            "user_id": profile.user_id,
            "model_name": getattr(self._model, "model_name", None),
            "generation_kwargs": dict(self._generation_kwargs),
            "identify_changes": parsed_json,
            "raw_response": raw_response,
        }

        tmp_path = output_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_path)
        except Exception as exc:  # noqa: BLE001
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"Failed to write identified changes artifact to {output_path}"
            ) from exc

        logger.info("Saved identified change options to %s", output_path)

    def _build_identify_changes_prompt(
        self,
        profile: UserProfile,
        requested_fields: Optional[List[str]] = None,
    ) -> str:
        """
        Render the identify-changes prompt, optionally restricting it to specific fields.

        Args:
            profile: User profile being personalized.
            requested_fields: Optional subset of fields to request in this call.

        Returns:
            str: Rendered identify-changes prompt.
        """
        prompt = self._identify_changes_template.format(
            user_profile_json=profile.model_dump_json(indent=2)
        )
        if not requested_fields:
            return prompt

        requested_fields_block = "\n".join(f"- {field}" for field in requested_fields)
        return (
            f"{prompt}\n\n"
            "CALL-SPECIFIC REQUIREMENT:\n"
            "For this call, output entries ONLY for the following fields and omit all others:\n"
            f"{requested_fields_block}\n"
            f'The "changes_by_field" array must contain exactly {len(requested_fields)} object(s), '
            "one per listed field.\n"
            "Do not merge fields, do not duplicate fields, and do not return any extra fields."
        )

    @staticmethod
    def _validate_identify_changes_fields(
        parsed_json: Dict[str, Any],
        requested_fields: List[str],
        *,
        context: str,
    ) -> Dict[str, Any]:
        """
        Validate that identify-changes output contains exactly the requested fields.

        Args:
            parsed_json: Parsed identify-changes payload.
            requested_fields: Fields expected in this response.
            context: Short label for debugging messages.

        Returns:
            Dict[str, Any]: The validated payload.

        Raises:
            ValueError: If the payload omits, duplicates, or adds fields.
        """
        changes_by_field = parsed_json.get("changes_by_field")
        if not isinstance(changes_by_field, list):
            raise ValueError(
                f"Identify-changes response for {context} is missing a list-valued "
                "'changes_by_field' entry."
            )

        actual_fields = []
        for item in changes_by_field:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Identify-changes response for {context} contains a non-object "
                    "entry in 'changes_by_field'."
                )
            field_name = item.get("field")
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError(
                    f"Identify-changes response for {context} contains an invalid "
                    "'field' value: {field_name!r}"
                )
            actual_fields.append(field_name)

        if len(actual_fields) != len(set(actual_fields)):
            raise ValueError(
                f"Identify-changes response for {context} contains duplicate fields: "
                f"{actual_fields}"
            )

        expected_set = set(requested_fields)
        actual_set = set(actual_fields)
        if actual_set != expected_set:
            missing_fields = sorted(expected_set - actual_set)
            extra_fields = sorted(actual_set - expected_set)
            raise ValueError(
                f"Identify-changes response for {context} returned the wrong field set. "
                f"Missing={missing_fields}, extra={extra_fields}, actual={actual_fields}"
            )

        return parsed_json

    def _recover_identify_change_options_by_field(
        self,
        profile: UserProfile,
        *,
        original_exception: Exception,
    ) -> tuple[Dict[str, Any], str]:
        """
        Retry identify-changes generation one field at a time after a bulk parse failure.

        Args:
            profile: User profile being personalized.
            original_exception: Exception from the original bulk parse attempt.

        Returns:
            tuple[Dict[str, Any], str]: Recovered parsed payload and debug raw-response blob.

        Raises:
            RuntimeError: If any field-scoped retry fails.
        """
        recovered_changes: List[Dict[str, Any]] = []
        recovery_payload: Dict[str, Any] = {
            "recovery_mode": "field_scoped_identify_changes",
            "initial_error": repr(original_exception),
            "responses": [],
        }

        for field_name in self._IDENTIFY_CHANGE_TARGET_FIELDS:
            prompt = self._build_identify_changes_prompt(profile, [field_name])
            response_str = self._model.generate(
                prompt,
                json_schema=self._identify_changes_schema,
                **self._generation_kwargs,
            )
            logger.debug(
                "Raw response for field-scoped change identification (%s):\n%s",
                field_name,
                response_str,
            )
            recovery_payload["responses"].append(
                {"requested_fields": [field_name], "raw_response": response_str}
            )

            try:
                parsed_json = parse_and_validate_json(
                    response_str, schema=self._identify_changes_schema
                )
                parsed_json = self._validate_identify_changes_fields(
                    parsed_json,
                    [field_name],
                    context=f"profile={profile.user_id}, field={field_name}",
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to recover identify-changes output after the initial bulk "
                    f"parse error for profile '{profile.user_id}'. "
                    f"Field-scoped retry failed for field '{field_name}'."
                ) from exc

            recovered_changes.extend(parsed_json["changes_by_field"])

        return (
            {"changes_by_field": recovered_changes},
            json.dumps(recovery_payload, indent=2, ensure_ascii=False),
        )

    def _select_compose_prompt_template(self, sample: BenchmarkSample) -> str:
        """
        Select the compose prompt template based on the benchmark.

        HumanEval+ personalization must be a prefix-only rewrite, which is later
        concatenated with the original prompt. This is controlled by a dedicated
        prompt template key: `compose_prompt_humaneval_plus`.
        """
        if sample.source_benchmark == "humaneval_plus":
            if not self._compose_prompt_template_humaneval_plus:
                raise ValueError(
                    "Model config missing personalization_prompts.compose_prompt_humaneval_plus "
                    "required for humaneval_plus."
                )
            return self._compose_prompt_template_humaneval_plus
        return self._compose_prompt_template

    @staticmethod
    def _finalize_modified_prompt(sample: BenchmarkSample, modified_prompt: str) -> str:
        """
        Finalize the stored modified prompt text for a benchmark.

        For HumanEval+, we store the final personalized prompt as:
            f\"{prefix}\\n{original_prompt}\"
        where `prefix` is what the personalization model generated.
        """
        if sample.source_benchmark != "humaneval_plus":
            return modified_prompt

        prefix = (modified_prompt or "").strip()
        original = (sample.prompt or "").strip()
        if not prefix:
            return original
        if not original:
            return prefix
        return f"{prefix}\n{original}"

    def _identify_change_options(
        self, profile: UserProfile
    ) -> ChangeIdentificationOutput:
        """Subtask 1: Identify possible changes based on the user profile."""
        logger.info(f"Identifying change options for profile: {profile.user_id}")

        prompt = self._build_identify_changes_prompt(profile)

        response_str = self._model.generate(
            prompt, json_schema=self._identify_changes_schema, **self._generation_kwargs
        )
        logger.debug(f"Raw response for change identification:\n{response_str}")

        raw_response_for_artifact = response_str
        try:
            parsed_json = parse_and_validate_json(
                response_str, schema=self._identify_changes_schema
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Bulk identify-changes response was not valid JSON/schema output for "
                "profile %s. Retrying with field-scoped requests.",
                profile.user_id,
                exc_info=exc,
            )
            parsed_json, raw_response_for_artifact = (
                self._recover_identify_change_options_by_field(
                    profile, original_exception=exc
                )
            )
        self._persist_identified_changes(
            profile=profile,
            parsed_json=parsed_json,
            raw_response=raw_response_for_artifact,
        )
        return ChangeIdentificationOutput(**parsed_json)

    def _compose_modified_prompts_batch(self, prompts: List[str]) -> List[str]:
        """
        Subtask 2: Compose modified prompts in batch.

        Args:
            prompts: Rendered prompt strings to send to the model.

        Returns:
            List[str]: Model outputs aligned with the input order.
        """
        requests = [
            GenerationRequest(
                prompt=p,
                json_schema=self._compose_prompt_schema,
                generation_kwargs=self._generation_kwargs,
            )
            for p in prompts
        ]
        raw_outputs = self._model.generate_batch(requests)
        parsed_outputs: List[str] = []
        for output in raw_outputs:
            parsed_outputs.append(
                self._unwrap_composed_output(output, self._compose_prompt_schema)
            )
        return parsed_outputs

    @staticmethod
    def _unwrap_composed_output(payload: Any, compose_schema: Dict[str, Any]) -> str:
        """
        Normalize the composed prompt output to a plain string.

        Args:
            payload: Raw model response for the composed prompt.
            compose_schema: JSON schema used for validation.

        Returns:
            The extracted modified prompt text, or the raw payload on failure.
        """
        if payload is None:
            return ""

        if isinstance(payload, dict):
            return str(payload.get("modified_prompt", ""))

        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped.startswith("{"):
                try:
                    parsed_json = parse_and_validate_json(
                        stripped, schema=compose_schema
                    )
                    return str(parsed_json.get("modified_prompt", stripped))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to parse composed prompt JSON; returning raw. Error: %s",
                        exc,
                    )
            return stripped

        return str(payload)

    def _verify_variations_batch(
        self, original_prompts: List[str], modified_prompts: List[str]
    ) -> List[VerificationOutput]:
        """
        Subtask 3: Verify multiple prompt variations in batch.
        """
        prompts = []
        for original_prompt, modified_prompt in zip(original_prompts, modified_prompts):
            prompts.append(
                self._verify_prompt_template.format(
                    original_prompt=original_prompt,
                    modified_prompt=modified_prompt,
                )
            )

        requests = [
            GenerationRequest(
                prompt=p,
                json_schema=self._verify_prompt_schema,
                generation_kwargs=self._generation_kwargs,
            )
            for p in prompts
        ]
        responses = self._model.generate_batch(requests)

        verified: List[VerificationOutput] = []
        for response_str in responses:
            parsed_json = parse_and_validate_json(
                response_str, schema=self._verify_prompt_schema
            )
            verified.append(VerificationOutput(**parsed_json))
        return verified

    def _verify_variation(
        self, original_prompt: str, modified_prompt: str
    ) -> VerificationOutput:
        """Subtask 3: Verify the modified prompt preserves the original's intent."""
        logger.info("Verifying prompt variation.")
        prompt = self._verify_prompt_template.format(
            original_prompt=original_prompt,
            modified_prompt=modified_prompt,
        )

        response_str = self._model.generate(
            prompt, json_schema=self._verify_prompt_schema, **self._generation_kwargs
        )
        logger.debug(f"Raw response for verification:\n{response_str}")

        parsed_json = parse_and_validate_json(
            response_str, schema=self._verify_prompt_schema
        )
        return VerificationOutput(**parsed_json)

    def build_dataset(
        self,
        samples: List[BenchmarkSample],
        profile: UserProfile,
        num_variations_per_sample: int,
    ) -> List[PersonalizedSample]:
        """
        Builds a full vibe-testing dataset for a given user profile.

        This orchestrates the three subtasks:
        1. Identifies all possible change options from the profile.
        2. For each sample, creates N unique compositions of changes.
        3. For each composition, generates and verifies a new prompt.

        Args:
            samples: A list of selected benchmark samples.
            profile: The user profile.
            num_variations_per_sample: The target number of variations for each sample.

        Returns:
            A list of PersonalizedSample objects, each containing the original
            sample and its generated variations.
        """
        change_options_output = self._identify_change_options(profile)
        options_by_field = change_options_output.changes_by_field

        personalized_samples: List[PersonalizedSample] = []
        # Go over each sample in the dataset with tqdm
        for sample in tqdm.tqdm(samples, desc="Processing samples", total=len(samples)):
            sample_id = sample.sample_id
            logger.info(f"--- Processing sample: {sample_id} ---")
            variations: List[PromptVariation] = []

            # Generate unique combinations of changes
            # This is a simple strategy: pick one change from each field category
            change_lists = [fc.options for fc in options_by_field]
            all_possible_combos = list(itertools.product(*change_lists))

            # If there are fewer possible combos than requested, just use all of them
            num_to_generate = min(num_variations_per_sample, len(all_possible_combos))

            # Randomly sample unique combinations
            selected_combos = random.sample(all_possible_combos, num_to_generate)

            # Prebuild prompts for composition
            compose_payloads = []
            test_list = sample.metadata.get("prompt_tests", {})
            plus_test_list = sample.metadata.get("tests", {}).get(
                "test", ""
            )  # this are the plus tests
            test_list = test_list  # + [plus_test_list]
            # prompt_with_tests = sample.prompt + format_test_cases_for_prompt(test_list)
            compose_template = self._select_compose_prompt_template(sample)

            for i, combo in enumerate(selected_combos):
                variation_id = f"{sample_id}-{profile.user_id}-var{i+1}"
                applied_change_options = list(combo)
                applied_change_names = [opt.name for opt in applied_change_options]
                changes_description = "\n".join(
                    f"- {c.name}: {c.change} (e.g., '{c.example}')"
                    for c in applied_change_options
                )
                # if compose_template has {evaluation_tests}, add it to the render_prompt
                if "{evaluation_tests}" in compose_template:
                    render_prompt = compose_template.format(
                        original_prompt=sample.prompt,  # prompt without tests
                        changes_description=changes_description,
                        evaluation_tests=format_test_cases_for_prompt(test_list),
                    )
                else:
                    render_prompt = compose_template.format(
                        original_prompt=sample.prompt,  # prompt without tests,
                        changes_description=changes_description,
                    )

                compose_payloads.append(
                    {
                        "variation_id": variation_id,
                        "applied_changes": applied_change_names,
                        "render_prompt": render_prompt,
                    }
                )

            # Compose in batches
            composed_outputs: List[str] = []
            for start in range(0, len(compose_payloads), self._batch_size):
                chunk = compose_payloads[start : start + self._batch_size]
                prompts = [item["render_prompt"] for item in chunk]
                composed_outputs.extend(self._compose_modified_prompts_batch(prompts))

            # Verify in batches, aligned to compose_payloads order
            verification_inputs = []
            for payload, modified_prompt in zip(compose_payloads, composed_outputs):
                finalized_prompt = self._finalize_modified_prompt(
                    sample, modified_prompt
                )
                verification_inputs.append(
                    {
                        "variation_id": payload["variation_id"],
                        "applied_changes": payload["applied_changes"],
                        "modified_prompt": finalized_prompt,
                    }
                )

            verifications: List[VerificationOutput] = []
            for start in range(0, len(verification_inputs), self._batch_size):
                chunk = verification_inputs[start : start + self._batch_size]
                original_prompts = [sample.prompt for _ in chunk]
                modified_prompts = [item["modified_prompt"] for item in chunk]
                verifications.extend(
                    self._verify_variations_batch(original_prompts, modified_prompts)
                )

            for payload, verification in zip(verification_inputs, verifications):
                variation = PromptVariation(
                    variation_id=payload["variation_id"],
                    applied_changes=payload["applied_changes"],
                    modified_prompt=payload["modified_prompt"],
                    verification=verification,
                )
                variations.append(variation)
                logger.info(
                    f"Successfully created and verified variation {payload['variation_id']}"
                )

            personalized_samples.append(
                PersonalizedSample(original_sample=sample, variations=variations)
            )
            logger.info(
                f"Finished processing sample {sample_id}, created {len(variations)} variations."
            )

        return personalized_samples

    def _validate_simple_personalization_config(self) -> None:
        """
        Validate that simple personalization prompts are configured.

        Raises:
            ValueError: If required simple personalization templates are missing.
        """
        if not self._simple_compose_template or not self._simple_verify_template:
            raise ValueError(
                "Simple personalization prompts not fully configured in model config. "
                "Ensure 'simple_personalization_prompts.compose_prompt' and "
                "'simple_personalization_prompts.verify_prompt' are set."
            )

    def _select_simple_compose_template(self, sample: BenchmarkSample) -> str:
        """
        Select the simple compose prompt template based on the benchmark.

        Args:
            sample: The benchmark sample to personalize.

        Returns:
            str: The appropriate simple compose template.

        Raises:
            ValueError: If HumanEval+ template is missing when required.
        """
        if sample.source_benchmark == "humaneval_plus":
            if not self._simple_compose_template_humaneval_plus:
                raise ValueError(
                    "Model config missing simple_personalization_prompts."
                    "compose_prompt_humaneval_plus required for humaneval_plus."
                )
            return self._simple_compose_template_humaneval_plus
        return self._simple_compose_template

    @staticmethod
    def _build_persona_summary(profile: UserProfile) -> str:
        """
        Build a concise persona summary string from a user profile.

        Args:
            profile: The user profile.

        Returns:
            str: A short persona summary suitable for simple personalization prompts.
        """
        parts = []
        persona_desc = getattr(profile, "persona_description", None)
        if persona_desc:
            parts.append(persona_desc)
        else:
            dump = profile.model_dump(exclude_none=True)
            desc = dump.get("persona", {}).get("description", "")
            if desc:
                parts.append(desc)
        if not parts:
            parts.append(f"User: {profile.user_id}")
        return " ".join(parts)

    @staticmethod
    def _build_changes_description(change_options: List[ChangeOption]) -> str:
        """
        Build a rendered changes description block from change options.

        Args:
            change_options: Change options to describe in the compose prompt.

        Returns:
            str: Human-readable bullet list of candidate prompt changes.

        Raises:
            ValueError: If no change options are provided.
        """
        if not change_options:
            raise ValueError("Cannot build changes_description from an empty change list.")
        return "\n".join(
            f"- {change.name}: {change.change} (e.g., '{change.example}')"
            for change in change_options
        )

    @staticmethod
    def _select_simple_change_options(
        options_by_field: List[FieldChanges],
        num_variations_per_sample: int,
    ) -> List[List[ChangeOption]]:
        """
        Select inspiration change options for simple personalization variations.

        Unlike full personalization, simple personalization should stay minimal.
        We therefore sample a single identified change option per variation and
        pass it as inspiration to the compose prompt.

        Args:
            options_by_field: Identified change options grouped by profile field.
            num_variations_per_sample: Number of simple variations to create.

        Returns:
            List[List[ChangeOption]]: One applied-change list per variation.

        Raises:
            ValueError: If no change options are available.
        """
        flat_options = [
            option
            for field_changes in options_by_field
            for option in field_changes.options
        ]
        if not flat_options:
            raise ValueError(
                "Simple personalization requires at least one identified change option, "
                "but none were produced for the user profile."
            )

        shuffled_options = flat_options[:]
        random.shuffle(shuffled_options)
        selected: List[List[ChangeOption]] = []
        for index in range(num_variations_per_sample):
            selected.append([shuffled_options[index % len(shuffled_options)]])
        return selected

    def build_simple_dataset(
        self,
        samples: List[BenchmarkSample],
        profile: UserProfile,
        num_variations_per_sample: int,
    ) -> List[PersonalizedSample]:
        """
        Build a simple-personalized vibe-testing dataset.

        Unlike ``build_dataset``, this skips the full change-option identification
        step and instead produces minimally-modified prompts (at most 3-4 extra
        words, no new constraints) that subtly reflect the user persona.

        Each variation carries ``prompt_type`` and ``variant_label`` set to
        ``"simple_personalized"`` so downstream stages can filter and aggregate
        them as a distinct prompt type.

        Args:
            samples: A list of selected benchmark samples.
            profile: The user profile.
            num_variations_per_sample: The target number of simple variations per sample.

        Returns:
            A list of PersonalizedSample objects with simple personalized variations.

        Raises:
            ValueError: If simple personalization prompts are not configured.
        """
        self._validate_simple_personalization_config()
        change_options_output = self._identify_change_options(profile)
        options_by_field = change_options_output.changes_by_field

        persona_summary = self._build_persona_summary(profile)
        logger.info(
            "Building simple personalized dataset for user %s with persona: %s",
            profile.user_id,
            persona_summary[:80],
        )

        personalized_samples: List[PersonalizedSample] = []

        for sample in tqdm.tqdm(
            samples, desc="Processing samples (simple)", total=len(samples)
        ):
            sample_id = sample.sample_id
            logger.info("--- Processing sample (simple personalization): %s ---", sample_id)
            variations: List[PromptVariation] = []

            compose_template = self._select_simple_compose_template(sample)
            selected_change_options = self._select_simple_change_options(
                options_by_field,
                num_variations_per_sample,
            )
            test_list = sample.metadata.get("prompt_tests", {})

            compose_payloads = []
            for i, applied_change_options in enumerate(selected_change_options):
                variation_id = f"{sample_id}-{profile.user_id}-simple_var{i + 1}"
                applied_change_names = [option.name for option in applied_change_options]
                changes_description = self._build_changes_description(
                    applied_change_options
                )

                if "{evaluation_tests}" in compose_template:
                    render_prompt = compose_template.format(
                        persona_summary=persona_summary,
                        original_prompt=sample.prompt,
                        changes_description=changes_description,
                        evaluation_tests=format_test_cases_for_prompt(test_list),
                    )
                else:
                    render_prompt = compose_template.format(
                        persona_summary=persona_summary,
                        original_prompt=sample.prompt,
                        changes_description=changes_description,
                    )

                compose_payloads.append(
                    {
                        "variation_id": variation_id,
                        "applied_changes": applied_change_names,
                        "render_prompt": render_prompt,
                    }
                )

            composed_outputs: List[str] = []
            for start in range(0, len(compose_payloads), self._batch_size):
                chunk = compose_payloads[start : start + self._batch_size]
                prompts = [item["render_prompt"] for item in chunk]
                composed_outputs.extend(self._compose_modified_prompts_batch(prompts))

            verification_inputs = []
            for payload, modified_prompt in zip(compose_payloads, composed_outputs):
                finalized_prompt = self._finalize_modified_prompt(
                    sample, modified_prompt
                )
                verification_inputs.append(
                    {
                        "variation_id": payload["variation_id"],
                        "applied_changes": payload["applied_changes"],
                        "modified_prompt": finalized_prompt,
                    }
                )

            verifications: List[VerificationOutput] = []
            saved_verify_template = self._verify_prompt_template
            try:
                self._verify_prompt_template = self._simple_verify_template
                for start in range(0, len(verification_inputs), self._batch_size):
                    chunk = verification_inputs[start : start + self._batch_size]
                    original_prompts = [sample.prompt for _ in chunk]
                    modified_prompts = [item["modified_prompt"] for item in chunk]
                    verifications.extend(
                        self._verify_variations_batch(original_prompts, modified_prompts)
                    )
            finally:
                self._verify_prompt_template = saved_verify_template

            for payload, verification in zip(verification_inputs, verifications):
                variation = PromptVariation(
                    variation_id=payload["variation_id"],
                    applied_changes=payload["applied_changes"],
                    modified_prompt=payload["modified_prompt"],
                    verification=verification,
                )
                variations.append(variation)
                logger.info(
                    "Created simple variation %s", payload["variation_id"]
                )

            personalized_samples.append(
                PersonalizedSample(original_sample=sample, variations=variations)
            )
            logger.info(
                "Finished simple personalization for sample %s, created %d variations.",
                sample_id,
                len(variations),
            )

        return personalized_samples
