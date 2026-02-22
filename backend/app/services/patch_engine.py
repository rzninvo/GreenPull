import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from google import genai
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger("greenpull")


class PatchResult(BaseModel):
    """Structured response from Gemini for a code patch."""
    patched_code: str = Field(
        description="The complete, fully runnable modified Python code. NO markdown formatting, NO explanations — just the code."
    )
    technique_used: str = Field(
        description="Short label: 'AMP', 'LoRA', or 'AMP+LoRA'"
    )
    changes_summary: str = Field(
        description="2-3 sentence summary of what was changed and why"
    )


@dataclass
class FilePatchResult:
    """Result of patching a single file."""
    file_path: str
    role: str
    optimization: str
    original_code: str
    patched_code: str
    diff: str
    changes_summary: str


@dataclass
class MultiPatchResult:
    """Aggregated result of patching multiple files."""
    file_patches: list[FilePatchResult]
    combined_diff: str
    optimization_plan: str


class PatchEngine:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def apply_patch(
        self,
        repo_path: Path,
        entrypoint_file: str,
        framework: str,
        patch_type: str,
        dep_context: str = "",
        config_context: str = "",
        imports_context: str = "",
        readme_context: str = "",
    ) -> Tuple[str, str]:
        file_path = repo_path / entrypoint_file
        original_code = file_path.read_text(errors="ignore")

        logger.info(f"[Patch] Applying '{patch_type}' patch to {entrypoint_file} (framework={framework})")
        logger.info(f"[Patch] Original code: {len(original_code)} chars, "
                     f"{len(original_code.splitlines())} lines")

        ctx = self._build_context(dep_context, config_context, imports_context, readme_context)

        if patch_type == "amp":
            patched_code = self._apply_amp(original_code, framework, entrypoint_file, ctx)
        elif patch_type == "lora":
            patched_code = self._apply_lora(original_code, framework, entrypoint_file, ctx)
        elif patch_type == "both":
            intermediate = self._apply_amp(original_code, framework, entrypoint_file, ctx)
            patched_code = self._apply_lora(intermediate, framework, entrypoint_file, ctx)
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")

        # Safety: if Gemini returned empty/garbage, keep original code
        if not patched_code or len(patched_code) < 20:
            logger.warning("[Patch] Gemini returned empty/short code, keeping original")
            patched_code = original_code

        logger.info(f"[Patch] Patched code: {len(patched_code)} chars, "
                     f"{len(patched_code.splitlines())} lines")

        # Don't write patched code to disk — preview only
        diff = "".join(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            patched_code.splitlines(keepends=True),
            fromfile="original",
            tofile="optimized",
        ))

        logger.info(f"[Patch] Diff:\n{diff[:3000]}")

        return patched_code, diff

    def _build_context(
        self, dep_context: str, config_context: str,
        imports_context: str, readme_context: str,
    ) -> str:
        """Build a combined context string from all repo context sources."""
        sections = []
        if dep_context:
            sections.append(f"DEPENDENCIES (requirements.txt / pyproject.toml):\n{dep_context}")
        if config_context:
            sections.append(f"CONFIG FILES FOUND IN REPO:\n{config_context}")
        if imports_context:
            sections.append(f"LOCAL MODULES IMPORTED BY ENTRYPOINT:\n{imports_context}")
        if readme_context:
            sections.append(f"README:\n{readme_context}")
        return "\n\n".join(sections)

    def _apply_amp(self, code: str, framework: str, filename: str, context: str,
                   extra_context: str = "") -> str:
        plan_section = f"\nOPTIMIZATION PLAN (coordinate with other files):\n{extra_context}\n" if extra_context else ""
        prompt = f"""You are modifying a machine learning training script to add Automatic Mixed Precision (AMP) for energy efficiency.

FRAMEWORK: {framework}
FILE: {filename}

{f"REPOSITORY CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}{plan_section}
ORIGINAL CODE:
```python
{code}
```

Apply AMP mixed precision training:

FOR PYTORCH:
- Add `from torch.cuda.amp import autocast, GradScaler`
- Wrap the forward pass in `with autocast():`
- Use GradScaler: `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`

FOR HUGGINGFACE TRANSFORMERS:
- Add `fp16=True` to TrainingArguments (or `bf16=True` for Ampere+)

FOR TENSORFLOW/KERAS:
- Add `from tensorflow.keras import mixed_precision`
- Add `mixed_precision.set_global_policy('mixed_float16')`

RULES:
- Return the complete modified Python code.
- Only use libraries that are in the dependencies. Do not add new dependencies.
- Preserve all original functionality.
- The code must be syntactically valid Python.
- If the code already uses AMP, return it unchanged."""

        logger.debug(f"[Patch] AMP prompt length: {len(prompt)} chars")
        result = self._call_gemini(prompt)
        return result.patched_code if result else code

    def _apply_lora(self, code: str, framework: str, filename: str, context: str,
                    extra_context: str = "") -> str:
        plan_section = f"\nOPTIMIZATION PLAN (coordinate with other files):\n{extra_context}\n" if extra_context else ""
        prompt = f"""You are modifying a machine learning training script to add LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

FRAMEWORK: {framework}
FILE: {filename}

{f"REPOSITORY CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}{plan_section}
ORIGINAL CODE:
```python
{code}
```

Apply LoRA:

FOR PYTORCH/HUGGINGFACE:
- Add `from peft import LoraConfig, get_peft_model`
- Create config: `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)`
- Wrap model: `model = get_peft_model(model, lora_config)`

If the model has no attention layers (CNN, MLP), return the code unchanged with a comment.

RULES:
- Return the complete modified Python code.
- Only use libraries that are in the dependencies. Do not add new dependencies.
- Preserve all original functionality.
- The code must be syntactically valid Python."""

        logger.debug(f"[Patch] LoRA prompt length: {len(prompt)} chars")
        result = self._call_gemini(prompt)
        return result.patched_code if result else code

    def _apply_dataloader_opts(self, code: str, framework: str, filename: str,
                                optimization_plan: str, dep_context: str) -> str:
        prompt = f"""You are optimizing a data loading module for energy efficiency.

FRAMEWORK: {framework}
FILE: {filename}
DEPENDENCIES: {dep_context[:1000]}

OPTIMIZATION PLAN:
{optimization_plan[:2000]}

ORIGINAL CODE:
```python
{code}
```

Apply these DataLoader optimizations where applicable:
- Add pin_memory=True for GPU training
- Set num_workers=4 (or appropriate count)
- Add prefetch_factor=2
- Add persistent_workers=True if num_workers > 0
- For tf.data: add .prefetch(tf.data.AUTOTUNE) and .cache()

RULES:
- Return the complete modified Python code.
- Only apply changes that make sense for this file.
- Preserve all original functionality.
- The code must be syntactically valid Python."""

        result = self._call_gemini(prompt)
        return result.patched_code if result else code

    def _apply_config_update(self, code: str, framework: str, filename: str,
                              optimization_plan: str, dep_context: str) -> str:
        prompt = f"""You are updating a training configuration file to reflect optimizations.

FRAMEWORK: {framework}
FILE: {filename}

OPTIMIZATION PLAN:
{optimization_plan[:2000]}

ORIGINAL CODE:
```python
{code}
```

Update the config to include:
- fp16=True or bf16=True in TrainingArguments (if HuggingFace)
- gradient_accumulation_steps if batch size was changed
- dataloader_num_workers, dataloader_pin_memory if applicable
- Any other settings mentioned in the optimization plan

RULES:
- Return the complete modified Python code.
- Preserve all original functionality.
- The code must be syntactically valid Python."""

        result = self._call_gemini(prompt)
        return result.patched_code if result else code

    # --- Multi-file patching ---

    def apply_multi_file_patch(
        self,
        patchable_files: list,
        framework: str,
        patch_type: str,
        dep_context: str = "",
        config_context: str = "",
        readme_context: str = "",
    ) -> MultiPatchResult:
        """
        Two-phase multi-file patching:
        Phase 1: Generate optimization plan across all files
        Phase 2: Patch each file individually with plan as context
        """
        logger.info(f"[Patch] Multi-file patch: {len(patchable_files)} files, type={patch_type}")

        # Phase 1: Planning call
        optimization_plan = self._generate_optimization_plan(
            patchable_files, framework, patch_type, dep_context
        )
        logger.info(f"[Patch] Optimization plan: {len(optimization_plan)} chars")

        # Phase 2: Patch each file
        ctx = self._build_context(dep_context, config_context, "", readme_context)
        file_patches: list[FilePatchResult] = []

        for pf in patchable_files:
            logger.info(f"[Patch] Patching {pf.path} (role={pf.role}, opt={pf.optimization})")
            original_code = pf.content

            if pf.optimization == "amp":
                patched_code = self._apply_amp(original_code, framework, pf.path, ctx,
                                                extra_context=optimization_plan)
            elif pf.optimization == "lora":
                patched_code = self._apply_lora(original_code, framework, pf.path, ctx,
                                                 extra_context=optimization_plan)
            elif pf.optimization == "amp+lora":
                intermediate = self._apply_amp(original_code, framework, pf.path, ctx,
                                                extra_context=optimization_plan)
                patched_code = self._apply_lora(intermediate, framework, pf.path, ctx,
                                                 extra_context=optimization_plan)
            elif pf.optimization == "dataloader_opts":
                patched_code = self._apply_dataloader_opts(
                    original_code, framework, pf.path, optimization_plan, dep_context)
            elif pf.optimization == "config_update":
                patched_code = self._apply_config_update(
                    original_code, framework, pf.path, optimization_plan, dep_context)
            else:
                continue

            if not patched_code or len(patched_code) < 20:
                patched_code = original_code

            diff = self._make_file_diff(pf.path, original_code, patched_code)
            if not diff.strip():
                logger.info(f"[Patch] No changes for {pf.path}, skipping")
                continue

            file_patches.append(FilePatchResult(
                file_path=pf.path,
                role=pf.role,
                optimization=pf.optimization,
                original_code=original_code,
                patched_code=patched_code,
                diff=diff,
                changes_summary=f"{pf.optimization} applied to {pf.role}",
            ))

        combined_diff = self._assemble_combined_diff(file_patches)
        logger.info(f"[Patch] Multi-file result: {len(file_patches)} files patched, "
                     f"{len(combined_diff)} chars of diff")

        return MultiPatchResult(
            file_patches=file_patches,
            combined_diff=combined_diff,
            optimization_plan=optimization_plan,
        )

    def _generate_optimization_plan(
        self, patchable_files: list, framework: str, patch_type: str, dep_context: str
    ) -> str:
        """Single Gemini call to produce a coordination plan across all files."""
        file_summaries = []
        for pf in patchable_files:
            truncated = "\n".join(pf.content.splitlines()[:100])
            file_summaries.append(
                f"FILE: {pf.path}\nROLE: {pf.role}\n"
                f"PLANNED OPTIMIZATION: {pf.optimization}\n"
                f"FIRST 100 LINES:\n```python\n{truncated}\n```"
            )

        sep = "---\n"
        prompt = f"""You are planning optimizations across multiple files in an ML training codebase.

FRAMEWORK: {framework}
PATCH TYPE: {patch_type}
DEPENDENCIES:
{dep_context[:2000]}

FILES TO OPTIMIZE:
{sep.join(file_summaries)}

Create a coordination plan. For each file, describe:
1. What specific changes to make
2. How changes relate to other files (imports, function signatures)
3. Any new imports needed

Be specific about variable names and function signatures. Keep the plan concise."""

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.1},
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"[Patch] Planning call failed: {e}")
            return ""

    @staticmethod
    def _make_file_diff(file_path: str, original: str, patched: str) -> str:
        """Generate a unified diff with git-style a/b headers."""
        return "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))

    @staticmethod
    def _assemble_combined_diff(file_patches: list[FilePatchResult]) -> str:
        """Concatenate all file diffs into one combined unified diff."""
        parts = [fp.diff for fp in file_patches if fp.diff.strip()]
        return "\n".join(parts)

    def _call_gemini(self, prompt: str) -> Optional[PatchResult]:
        """Call Gemini with structured output, with fallback to text parsing."""
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PatchResult,
                    "temperature": 0.1,
                },
            )

            logger.debug(f"[Patch] Gemini response received ({len(response.text or '')} chars)")

            # Try structured output first
            if response.parsed:
                return response.parsed

            # Fallback: parse text as JSON
            logger.warning("[Patch] Gemini structured output was None, trying text parse")
            return self._parse_text_fallback(response.text)

        except Exception as e:
            logger.error(f"[Patch] Gemini call failed: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _parse_text_fallback(text: str) -> Optional[PatchResult]:
        """Try to extract a PatchResult from raw text if structured output failed."""
        if not text:
            return None
        text = text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return PatchResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Last resort: if the text looks like raw Python code, wrap it
            if "import " in text or "def " in text or "class " in text:
                return PatchResult(
                    patched_code=text,
                    technique_used="unknown",
                    changes_summary="Extracted from raw response",
                )
            return None
