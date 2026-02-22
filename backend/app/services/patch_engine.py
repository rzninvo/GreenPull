import difflib
import json
import logging
import re
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

    def _apply_amp(self, code: str, framework: str, filename: str, context: str) -> str:
        prompt = f"""You are modifying a machine learning training script to add Automatic Mixed Precision (AMP) for energy efficiency.

FRAMEWORK: {framework}
FILE: {filename}

{f"REPOSITORY CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}
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

    def _apply_lora(self, code: str, framework: str, filename: str, context: str) -> str:
        prompt = f"""You are modifying a machine learning training script to add LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

FRAMEWORK: {framework}
FILE: {filename}

{f"REPOSITORY CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}
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
