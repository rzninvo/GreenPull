import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import git
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger("greenpull")

# --- Regex patterns for heuristic scoring ---

POSITIVE_PATTERNS = [
    # (regex, points, label)
    (r"from\s+transformers\s+import\s+.*Trainer", 50, "hf_trainer_import"),
    (r"Trainer\s*\(", 50, "trainer_call"),
    (r"TrainingArguments\s*\(", 40, "training_args"),
    (r"trainer\.train\s*\(", 40, "trainer_train"),
    (r"from\s+pytorch_lightning", 45, "lightning_import"),
    (r"trainer\.fit\s*\(", 45, "lightning_fit"),
    (r"loss\.backward\s*\(", 40, "backward"),
    (r"optimizer\.step\s*\(", 40, "optimizer_step"),
    (r"model\.train\s*\(", 20, "model_train"),
    (r"model\.fit\s*\(", 40, "keras_fit"),
    (r"model\.compile\s*\(", 30, "keras_compile"),
    (r"from\s+(tensorflow|keras)", 20, "tf_import"),
    (r"\.fit\s*\(\s*X", 35, "sklearn_fit"),
    (r"add_argument.*--epochs", 20, "arg_epochs"),
    (r"add_argument.*--(lr|learning.rate)", 20, "arg_lr"),
    (r"add_argument.*--batch", 20, "arg_batch"),
    (r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', 15, "main_block"),
    (r"for\s+.*epoch.*in\s+range", 30, "epoch_loop"),
    (r"torch\.cuda\.amp", 25, "torch_amp"),
    (r"accelerate", 20, "accelerate"),
    (r"from\s+peft\s+import", 25, "peft_import"),
]

NEGATIVE_PATTERNS = [
    (r"model\.eval\s*\(", -15, "eval_only"),
    (r"torch\.no_grad\s*\(", -10, "no_grad"),
    (r"model\.predict\s*\(", -10, "predict_only"),
]

# Files/dirs to skip during scanning
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".tox", ".eggs", "dist", "build", ".mypy_cache",
}


@dataclass
class CandidateScript:
    path: str  # relative to repo root
    score: int = 0
    patterns: list[str] = field(default_factory=list)
    content: str = ""


@dataclass
class DetectionResult:
    entrypoint_file: Optional[str] = None
    run_command: Optional[str] = None
    framework: Optional[str] = None
    reasoning: str = ""
    confidence: str = "low"


class RepoAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.clone_dir = settings.CLONE_DIR

    def clone_repo(self, repo_url: str, job_id: str) -> Path:
        dest = self.clone_dir / job_id
        if dest.exists():
            shutil.rmtree(dest)
        logger.info(f"[Clone] Cloning {repo_url} -> {dest}")
        git.Repo.clone_from(repo_url, str(dest), depth=1)
        logger.info(f"[Clone] Done")
        return dest

    # ---- Phase B: Regex/heuristic scan ----

    def scan_for_candidates(self, repo_path: Path) -> list[CandidateScript]:
        logger.info(f"[Scan] Scanning {repo_path} for training scripts...")
        candidates = []
        total_py = 0

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                if not fname.endswith(".py"):
                    continue

                total_py += 1
                full_path = Path(root) / fname
                rel_path = str(full_path.relative_to(repo_path))

                try:
                    content = full_path.read_text(errors="ignore")
                except Exception:
                    continue

                if len(content.strip()) < 50:
                    continue

                score = 0
                patterns_found = []

                # Filename bonus
                lower_name = fname.lower()
                if lower_name in ("train.py", "training.py", "run_training.py", "run_train.py"):
                    score += 40
                    patterns_found.append("training_filename")
                elif "train" in lower_name:
                    score += 25
                    patterns_found.append("train_in_name")
                elif lower_name == "finetune.py" or "fine_tune" in lower_name:
                    score += 35
                    patterns_found.append("finetune_filename")

                for pattern, points, label in POSITIVE_PATTERNS:
                    if re.search(pattern, content):
                        score += points
                        patterns_found.append(label)

                for pattern, points, label in NEGATIVE_PATTERNS:
                    if re.search(pattern, content):
                        score += points
                        patterns_found.append(label)

                if score > 30:
                    candidates.append(CandidateScript(
                        path=rel_path,
                        score=score,
                        patterns=patterns_found,
                        content=content,
                    ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:10]

        logger.info(f"[Scan] Scanned {total_py} .py files, found {len(candidates)} candidates:")
        for c in candidates:
            logger.info(f"  {c.path:40s}  score={c.score:4d}  patterns={c.patterns}")

        return candidates

    # ---- Phase C: Iterative GPT analysis ----

    def analyze_scripts_iteratively(
        self, candidates: list[CandidateScript], repo_path: Path
    ) -> DetectionResult:
        if not candidates:
            logger.warning("[Detect] No candidates found by heuristic scan.")
            return DetectionResult(reasoning="No candidate training scripts found by heuristic scan.")

        dep_context = self._get_dependency_context(repo_path)
        logger.debug(f"[Detect] Dependency context:\n{dep_context[:500]}")

        accumulated_summaries = ""
        best_result = None

        for i, script in enumerate(candidates, 1):
            logger.info(f"[Detect] Analyzing script {i}/{len(candidates)}: {script.path} (score={script.score})")

            content = script.content
            lines = content.splitlines()
            if len(lines) > 500:
                content = "\n".join(lines[:500]) + "\n# ... (truncated)"

            prompt = f"""You are analyzing a machine learning repository to find the main training pipeline.

DEPENDENCY CONTEXT:
{dep_context}

SUMMARIES OF PREVIOUSLY ANALYZED SCRIPTS:
{accumulated_summaries if accumulated_summaries else "(none yet — this is the first script)"}

NOW ANALYZE THIS SCRIPT:
File: {script.path}
Heuristic score: {script.score}
Detected patterns: {', '.join(script.patterns)}

```python
{content}
```

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "summary": "2-3 sentence summary of what this script does",
  "is_main_entrypoint": "yes" or "no" or "partial",
  "run_command": "python {script.path} --epochs 1" or null if not main entrypoint,
  "framework": "pytorch" or "tensorflow" or "huggingface" or "jax" or "sklearn" or "unknown",
  "confidence": "high" or "medium" or "low"
}}

IMPORTANT: If this is the main training script, set run_command to the minimal command that runs training for 1 epoch or a short run. Override epochs/steps to 1."""

            logger.debug(f"[Detect] GPT prompt length: {len(prompt)} chars")

            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0,
            )

            raw_response = response.choices[0].message.content
            logger.info(f"[Detect] GPT response for {script.path}:\n{raw_response}")

            result = self._parse_gpt_json(raw_response)
            if result is None:
                logger.warning(f"[Detect] Failed to parse GPT JSON for {script.path}")
                continue

            summary = result.get("summary", "Could not summarize.")
            is_main = result.get("is_main_entrypoint", "no")
            confidence = result.get("confidence", "low")
            accumulated_summaries += f"\n- {script.path} (score={script.score}): {summary}"

            logger.info(f"[Detect] -> is_main={is_main}, confidence={confidence}, "
                        f"framework={result.get('framework')}")

            # Check for early exit
            if is_main == "yes" and confidence == "high":
                logger.info(f"[Detect] HIGH CONFIDENCE MATCH: {script.path}")
                return DetectionResult(
                    entrypoint_file=script.path,
                    run_command=result.get("run_command"),
                    framework=result.get("framework", "unknown"),
                    reasoning=accumulated_summaries,
                    confidence="high",
                )

            # Track best candidate so far
            if is_main in ("yes", "partial"):
                conf_rank = {"high": 3, "medium": 2, "low": 1}
                entry_rank = {"yes": 2, "partial": 1}
                new_score = (
                    entry_rank.get(is_main, 0),
                    conf_rank.get(confidence, 0),
                )
                if best_result is None:
                    old_score = (0, 0)
                else:
                    old_score = (
                        entry_rank.get("yes" if best_result.entrypoint_file else "partial", 0),
                        conf_rank.get(best_result.confidence, 0),
                    )
                if new_score > old_score:
                    best_result = DetectionResult(
                        entrypoint_file=script.path,
                        run_command=result.get("run_command"),
                        framework=result.get("framework", "unknown"),
                        reasoning=accumulated_summaries,
                        confidence=confidence,
                    )

        if best_result and best_result.entrypoint_file:
            logger.info(f"[Detect] Best candidate: {best_result.entrypoint_file} "
                        f"(confidence={best_result.confidence})")
            best_result.reasoning = accumulated_summaries
            return best_result

        logger.info("[Detect] No clear winner, running final synthesis...")
        return self._final_synthesis(accumulated_summaries, candidates)

    def _final_synthesis(
        self, accumulated_summaries: str, candidates: list[CandidateScript]
    ) -> DetectionResult:
        prompt = f"""Based on analyzing all scripts in this ML repository, determine the main training entrypoint.

ALL SCRIPT SUMMARIES:
{accumulated_summaries}

AVAILABLE FILES (by heuristic score):
{chr(10).join(f'- {c.path} (score={c.score}, patterns={c.patterns})' for c in candidates)}

Respond with ONLY valid JSON:
{{
  "entrypoint_file": "path/to/main_training_script.py",
  "run_command": "python path/to/script.py --epochs 1",
  "framework": "pytorch" or "tensorflow" or "huggingface" or "jax" or "sklearn" or "unknown",
  "reasoning": "brief explanation of why this is the main entrypoint"
}}"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0,
        )

        raw = response.choices[0].message.content
        logger.info(f"[Detect] Final synthesis GPT response:\n{raw}")

        result = self._parse_gpt_json(raw)
        if result is None:
            logger.error("[Detect] Final synthesis JSON parse failed")
            return DetectionResult(reasoning="GPT could not determine the entrypoint.\n" + accumulated_summaries)

        return DetectionResult(
            entrypoint_file=result.get("entrypoint_file"),
            run_command=result.get("run_command"),
            framework=result.get("framework", "unknown"),
            reasoning=accumulated_summaries + "\n\nFinal decision: " + result.get("reasoning", ""),
            confidence="medium",
        )

    # ---- Helpers ----

    def _get_dependency_context(self, repo_path: Path) -> str:
        context_parts = []
        for meta_file in ["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"]:
            meta_path = repo_path / meta_file
            if meta_path.exists():
                content = meta_path.read_text(errors="ignore")[:2000]
                context_parts.append(f"--- {meta_file} ---\n{content}")
        return "\n".join(context_parts) if context_parts else "(no dependency files found)"

    @staticmethod
    def _parse_gpt_json(text: str) -> Optional[dict]:
        if text is None:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None
