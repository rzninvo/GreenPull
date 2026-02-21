import difflib
import logging
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger("greenpull")


class PatchEngine:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def apply_patch(
        self,
        repo_path: Path,
        entrypoint_file: str,
        framework: str,
        patch_type: str,
    ) -> Tuple[str, str]:
        file_path = repo_path / entrypoint_file
        original_code = file_path.read_text()

        logger.info(f"[Patch] Applying '{patch_type}' patch to {entrypoint_file} (framework={framework})")
        logger.info(f"[Patch] Original code: {len(original_code)} chars, "
                     f"{len(original_code.splitlines())} lines")

        if patch_type == "amp":
            patched_code = self._apply_amp(original_code, framework, entrypoint_file)
        elif patch_type == "lora":
            patched_code = self._apply_lora(original_code, framework, entrypoint_file)
        elif patch_type == "both":
            intermediate = self._apply_amp(original_code, framework, entrypoint_file)
            patched_code = self._apply_lora(intermediate, framework, entrypoint_file)
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")

        # Safety: if GPT returned empty/garbage, keep original code
        if not patched_code or len(patched_code) < 20:
            logger.warning("[Patch] GPT returned empty/short code, keeping original")
            patched_code = original_code

        logger.info(f"[Patch] Patched code: {len(patched_code)} chars, "
                     f"{len(patched_code.splitlines())} lines")

        file_path.write_text(patched_code)

        diff = "".join(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            patched_code.splitlines(keepends=True),
            fromfile="original",
            tofile="optimized",
        ))

        logger.info(f"[Patch] Diff:\n{diff[:3000]}")

        return patched_code, diff

    def _apply_amp(self, code: str, framework: str, filename: str) -> str:
        prompt = f"""You are modifying a machine learning training script to add Automatic Mixed Precision (AMP) for energy efficiency.

FRAMEWORK: {framework}
FILE: {filename}

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
- Return ONLY the complete modified Python code. No explanations, no markdown.
- Preserve all original functionality.
- The code must be syntactically valid Python."""

        logger.debug(f"[Patch] AMP prompt length: {len(prompt)} chars")
        return self._call_gpt(prompt)

    def _apply_lora(self, code: str, framework: str, filename: str) -> str:
        prompt = f"""You are modifying a machine learning training script to add LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

FRAMEWORK: {framework}
FILE: {filename}

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
- Return ONLY the complete modified Python code. No explanations, no markdown.
- Preserve all original functionality.
- The code must be syntactically valid Python."""

        logger.debug(f"[Patch] LoRA prompt length: {len(prompt)} chars")
        return self._call_gpt(prompt)

    def _call_gpt(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        text = text.strip()

        logger.debug(f"[Patch] GPT raw response ({len(text)} chars):\n{text[:1000]}")

        # Extract code from markdown block if GPT wrapped it
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        return text.strip()
