import ast
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

# --- Patterns for multi-file role classification ---

MODEL_PATTERNS = [
    (r"class\s+\w+\s*\(.*nn\.Module\)", 50, "nn_module"),
    (r"class\s+\w+\s*\(.*PreTrainedModel\)", 50, "pretrained_model"),
    (r"class\s+\w+\s*\(.*tf\.keras\.Model\)", 50, "keras_model"),
    (r"nn\.Linear|nn\.Conv\d?d|nn\.MultiheadAttention", 30, "nn_layers"),
    (r"self\.attention|self\.self_attn|self\.q_proj|self\.v_proj", 40, "attention"),
    (r"nn\.TransformerEncoder|nn\.TransformerDecoder", 35, "transformer"),
    (r"def\s+forward\s*\(self", 40, "forward_method"),
]

DATALOADER_PATTERNS = [
    (r"DataLoader\s*\(", 50, "dataloader_init"),
    (r"Dataset\s*\(|torch\.utils\.data", 40, "dataset"),
    (r"tf\.data\.Dataset", 40, "tf_dataset"),
    (r"collate_fn", 25, "collate_fn"),
    (r"class\s+\w+\s*\(.*Dataset\)", 45, "dataset_class"),
]

CONFIG_PATTERNS = [
    (r"TrainingArguments\s*\(", 60, "training_args"),
    (r"@dataclass[\s\S]{0,50}class\s+\w*[Cc]onfig", 40, "config_dataclass"),
    (r"num_train_epochs|learning_rate|per_device_train_batch_size", 35, "hf_config"),
    (r"argparse\.ArgumentParser", 30, "argparse"),
]

# Maps (role, patch_type) → optimization to apply
ROLE_OPTIMIZATION_MAP = {
    ("entrypoint", "amp"): "amp",
    ("entrypoint", "lora"): "lora",
    ("entrypoint", "both"): "amp+lora",
    ("model", "lora"): "lora",
    ("model", "both"): "lora",
    ("dataloader", "amp"): "dataloader_opts",
    ("dataloader", "lora"): "dataloader_opts",
    ("dataloader", "both"): "dataloader_opts",
    ("config", "amp"): "config_update",
    ("config", "lora"): "config_update",
    ("config", "both"): "config_update",
}

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


@dataclass
class PatchableFile:
    """A file identified as a candidate for optimization."""
    path: str          # relative to repo root
    role: str          # "entrypoint" | "model" | "dataloader" | "config"
    optimization: str  # "amp" | "lora" | "dataloader_opts" | "config_update" | "amp+lora"
    content: str       # file contents
    reason: str        # why this file was selected


class RepoAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.clone_dir = settings.CLONE_DIR

    @staticmethod
    def _normalize_github_url(url: str) -> str:
        """Convert browser GitHub URLs to clone-able URLs.
        e.g. https://github.com/user/repo/tree/main -> https://github.com/user/repo
        """
        url = url.rstrip("/")
        # Strip /tree/... or /blob/... suffixes
        url = re.sub(r"/(tree|blob)/[^/]+.*$", "", url)
        # Ensure .git suffix works too
        return url

    def clone_repo(self, repo_url: str, job_id: str) -> Path:
        dest = self.clone_dir / job_id
        if dest.exists():
            shutil.rmtree(dest)
        clone_url = self._normalize_github_url(repo_url)
        logger.info(f"[Clone] URL normalized: {repo_url} -> {clone_url}")
        logger.info(f"[Clone] Cloning into {dest}")
        git.Repo.clone_from(clone_url, str(dest), depth=1)
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
                max_completion_tokens=512,
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
            max_completion_tokens=512,
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

    # ---- Training config extraction ----

    def extract_training_config(
        self, code: str, entrypoint_file: str, framework: str,
        dep_context: str, config_context: str = "",
        imports_context: str = "", readme_context: str = "",
    ) -> dict:
        """Ask GPT to extract training configuration from source code + repo context."""
        lines = code.splitlines()
        if len(lines) > 600:
            code = "\n".join(lines[:600]) + "\n# ... (truncated)"

        # Build optional context sections
        extra_sections = ""
        if config_context:
            extra_sections += f"\nCONFIG FILES FOUND IN REPO:\n{config_context}\n"
        if imports_context:
            extra_sections += f"\nLOCAL MODULES IMPORTED BY ENTRYPOINT:\n{imports_context}\n"
        if readme_context:
            extra_sections += f"\nREADME:\n{readme_context}\n"

        prompt = f"""You are an expert ML engineer analyzing a training script to estimate its computational requirements.

DEPENDENCY CONTEXT:
{dep_context}
{extra_sections}
FRAMEWORK: {framework}
FILE: {entrypoint_file}

SOURCE CODE:
```python
{code}
```

Analyze this training script AND all the context above (config files, imported modules, README) to extract/estimate the following. Use config files for exact values when available. For values not found anywhere, make reasonable estimates.

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "model_type": "transformer" | "cnn" | "mlp" | "rnn" | "gnn" | "diffusion" | "gan" | "other",
  "model_name": "e.g. bert-base-uncased, resnet50, gpt2" or null,
  "parameter_count_millions": <float>,
  "epochs": <int>,
  "batch_size": <int>,
  "dataset_size_estimate": "small (<10K samples)" | "medium (10K-1M)" | "large (>1M)",
  "estimated_runtime_hours": <float, wall-clock time for full training>,
  "gpu_required": <bool>,
  "gpu_type": "A100" | "V100" | "T4" | "RTX_3090" | "H100" | etc.,
  "num_gpus": <int>,
  "cpu_type": "server_generic" | "cloud_generic" | "desktop_generic",
  "num_cpu_cores": <int>,
  "memory_gb": <float>,
  "reasoning": "2-3 sentences explaining your estimates"
}}

GUIDELINES:
- HuggingFace Trainer + BERT-base (110M), 3 epochs medium dataset: ~0.5-2h on V100
- PyTorch CNN ResNet50 (25M), 10 epochs ImageNet: ~10-20h on V100
- sklearn: minutes, CPU only
- Fine-tuning >1B params: hours on A100
- Use epochs/batch_size from code args if present"""

        logger.info(f"[Config] Extracting training config from {entrypoint_file}")

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            temperature=0,
        )

        raw = response.choices[0].message.content
        logger.info(f"[Config] GPT response:\n{raw}")

        result = self._parse_gpt_json(raw)
        if result is None:
            logger.error("[Config] Failed to parse, using defaults")
            return {
                "model_type": "unknown", "model_name": None,
                "parameter_count_millions": 100.0, "epochs": 3,
                "batch_size": 32, "dataset_size_estimate": "medium (10K-1M)",
                "estimated_runtime_hours": 1.0, "gpu_required": True,
                "gpu_type": "V100", "num_gpus": 1,
                "cpu_type": "server_generic", "num_cpu_cores": 8,
                "memory_gb": 16.0, "reasoning": "Defaults (GPT parse failed).",
            }
        return result

    def estimate_optimized_config(
        self, baseline_config: dict, patch_type: str, patch_diff: str,
        original_code: str, patched_code: str,
    ) -> dict:
        """Ask GPT how the optimization patch changes runtime/power."""
        prompt = f"""You are an expert ML engineer estimating how an optimization patch affects training.

BASELINE CONFIG:
{json.dumps(baseline_config, indent=2)}

PATCH TYPE: {patch_type}

PATCH DIFF:
```diff
{patch_diff[:3000]}
```

OPTIMIZED CODE (first 400 lines):
```python
{chr(10).join(patched_code.splitlines()[:400])}
```

AMP: 15-40% runtime reduction on GPU (Ampere+ GPUs more). No benefit CPU-only.
LoRA: 90-99% fewer trainable params, 20-40% faster. Only for attention-based models.
BOTH: 35-60% combined speedup.

Respond with ONLY valid JSON:
{{
  "estimated_runtime_hours": <float>,
  "runtime_reduction_pct": <float>,
  "memory_gb": <float>,
  "batch_size": <int>,
  "parameter_count_millions": <float, trainable params after optimization>,
  "reasoning": "2-3 sentences"
}}"""

        logger.info(f"[OptConfig] Estimating optimized config for {patch_type}")

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=512,
            temperature=0,
        )

        raw = response.choices[0].message.content
        logger.info(f"[OptConfig] GPT response:\n{raw}")

        result = self._parse_gpt_json(raw)
        if result is None:
            logger.warning("[OptConfig] Failed to parse, using 20% default reduction")
            return {
                "estimated_runtime_hours": baseline_config.get("estimated_runtime_hours", 1.0) * 0.8,
                "runtime_reduction_pct": 20.0,
                "memory_gb": baseline_config.get("memory_gb", 16.0) * 0.8,
                "batch_size": baseline_config.get("batch_size", 32),
                "parameter_count_millions": baseline_config.get("parameter_count_millions", 100.0),
                "reasoning": "Defaults (GPT parse failed). Assuming 20% improvement.",
            }
        return result

    # ---- Context gathering helpers ----

    def _gather_config_files(self, repo_path: Path) -> str:
        """Find and read config files (YAML, JSON, TOML, CFG, INI) in the repo."""
        config_extensions = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".conf"}
        # Also match common config filenames without extension
        config_names = {"config", "hparams", "hyperparameters", "params", "settings"}
        found = []
        total_chars = 0
        max_total = 6000  # cap total config context

        for fpath in sorted(repo_path.rglob("*")):
            if not fpath.is_file():
                continue
            # Skip hidden dirs and common junk
            parts = fpath.relative_to(repo_path).parts
            if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                continue
            # Skip package metadata and non-ML config files
            skip_names = {
                "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
                "package.json", "package-lock.json", "tsconfig.json",
                "babel.config.json", ".eslintrc.json", ".prettierrc.json",
                "Cargo.toml", "Cargo.lock", "composer.json",
            }
            if fpath.name in skip_names:
                continue

            stem_lower = fpath.stem.lower()
            ext = fpath.suffix.lower()

            is_config = (
                ext in config_extensions
                or any(cn in stem_lower for cn in config_names)
            )
            if not is_config:
                continue
            # Skip large files (likely data, not config)
            try:
                size = fpath.stat().st_size
                if size > 50_000:
                    continue
            except OSError:
                continue

            try:
                content = fpath.read_text(errors="ignore")[:2000]
            except Exception:
                continue

            rel = str(fpath.relative_to(repo_path))
            entry = f"--- {rel} ---\n{content}"
            if total_chars + len(entry) > max_total:
                break
            found.append(entry)
            total_chars += len(entry)

        return "\n".join(found) if found else ""

    # ---- Multi-file detection ----

    def _resolve_local_import_files(
        self, repo_path: Path, entrypoint_file: str
    ) -> list[tuple[str, str]]:
        """Return (relative_path, content) pairs for locally imported modules."""
        entrypoint_path = repo_path / entrypoint_file
        try:
            source = entrypoint_path.read_text(errors="ignore")
            tree = ast.parse(source)
        except Exception:
            return []

        local_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    local_modules.add(node.module.split(".")[0])

        results: list[tuple[str, str]] = []
        ep_dir = entrypoint_path.parent

        for mod_name in sorted(local_modules):
            candidates = [
                repo_path / f"{mod_name}.py",
                repo_path / mod_name / "__init__.py",
                repo_path / mod_name / "model.py",
                repo_path / mod_name / "models.py",
                repo_path / mod_name / "network.py",
                repo_path / mod_name / "net.py",
                repo_path / mod_name / "config.py",
                repo_path / mod_name / "data.py",
                repo_path / mod_name / "dataset.py",
                repo_path / mod_name / "dataloader.py",
            ]
            if ep_dir != repo_path:
                candidates.extend([
                    ep_dir / f"{mod_name}.py",
                    ep_dir / mod_name / "__init__.py",
                    ep_dir / mod_name / "model.py",
                    ep_dir / mod_name / "models.py",
                ])

            for cand in candidates:
                if cand.exists() and cand.is_file():
                    try:
                        content = cand.read_text(errors="ignore")
                    except Exception:
                        continue
                    rel = str(cand.relative_to(repo_path))
                    if rel == entrypoint_file:
                        continue
                    results.append((rel, content))
                    break
        return results

    @staticmethod
    def _classify_file(content: str) -> tuple[str, int, str]:
        """Classify a file by role using pattern scoring. Returns (role, score, reason)."""
        scores: dict[str, tuple[int, list[str]]] = {
            "model": (0, []),
            "dataloader": (0, []),
            "config": (0, []),
        }
        for pattern, points, label in MODEL_PATTERNS:
            if re.search(pattern, content):
                scores["model"] = (scores["model"][0] + points, scores["model"][1] + [label])
        for pattern, points, label in DATALOADER_PATTERNS:
            if re.search(pattern, content):
                scores["dataloader"] = (scores["dataloader"][0] + points, scores["dataloader"][1] + [label])
        for pattern, points, label in CONFIG_PATTERNS:
            if re.search(pattern, content):
                scores["config"] = (scores["config"][0] + points, scores["config"][1] + [label])

        best_role = max(scores, key=lambda r: scores[r][0])
        best_score = scores[best_role][0]
        reason = ", ".join(scores[best_role][1]) if scores[best_role][1] else "low confidence"
        return best_role, best_score, reason

    def identify_patchable_files(
        self,
        repo_path: Path,
        entrypoint_file: str,
        framework: str,
        patch_type: str,
        max_files: int = 5,
    ) -> list[PatchableFile]:
        """
        Identify files in the repo that can benefit from optimization.
        Always includes the entrypoint. Also checks local imports for
        model definitions, data loaders, and config files.
        """
        entrypoint_path = repo_path / entrypoint_file
        entrypoint_content = entrypoint_path.read_text(errors="ignore")

        # Determine entrypoint optimization
        ep_opt = ROLE_OPTIMIZATION_MAP.get(("entrypoint", patch_type), patch_type)
        result = [PatchableFile(
            path=entrypoint_file,
            role="entrypoint",
            optimization=ep_opt,
            content=entrypoint_content,
            reason="main training script",
        )]

        # Resolve local imports and classify each
        import_files = self._resolve_local_import_files(repo_path, entrypoint_file)
        for rel_path, content in import_files:
            if len(result) >= max_files:
                break

            role, score, reason = self._classify_file(content)

            if score < 30:
                continue

            opt = ROLE_OPTIMIZATION_MAP.get((role, patch_type))
            if not opt:
                continue

            result.append(PatchableFile(
                path=rel_path,
                role=role,
                optimization=opt,
                content=content,
                reason=reason,
            ))

        logger.info(f"[Analyzer] Identified {len(result)} patchable files")
        for pf in result:
            logger.info(f"  {pf.path} role={pf.role} opt={pf.optimization} ({pf.reason})")

        return result

    def _resolve_local_imports(self, repo_path: Path, entrypoint_file: str) -> str:
        """Parse the entrypoint's imports and read locally-defined modules."""
        entrypoint_path = repo_path / entrypoint_file
        try:
            source = entrypoint_path.read_text(errors="ignore")
            tree = ast.parse(source)
        except Exception:
            return ""

        local_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    local_modules.add(node.module.split(".")[0])

        # Check which imports correspond to local files/dirs in the repo
        found = []
        total_chars = 0
        max_total = 8000

        for mod_name in sorted(local_modules):
            # Try mod_name.py or mod_name/__init__.py
            candidates = [
                repo_path / f"{mod_name}.py",
                repo_path / mod_name / "__init__.py",
                repo_path / mod_name / "model.py",
                repo_path / mod_name / "models.py",
                repo_path / mod_name / "network.py",
                repo_path / mod_name / "net.py",
                repo_path / mod_name / "config.py",
            ]
            # Also check relative to entrypoint's directory
            ep_dir = entrypoint_path.parent
            if ep_dir != repo_path:
                candidates.extend([
                    ep_dir / f"{mod_name}.py",
                    ep_dir / mod_name / "__init__.py",
                    ep_dir / mod_name / "model.py",
                    ep_dir / mod_name / "models.py",
                ])

            for cand in candidates:
                if cand.exists() and cand.is_file():
                    try:
                        content = cand.read_text(errors="ignore")
                        lines = content.splitlines()
                        if len(lines) > 200:
                            content = "\n".join(lines[:200]) + "\n# ... (truncated)"
                    except Exception:
                        continue

                    rel = str(cand.relative_to(repo_path))
                    # Skip the entrypoint itself
                    if rel == entrypoint_file:
                        continue
                    entry = f"--- {rel} ---\n{content}"
                    if total_chars + len(entry) > max_total:
                        break
                    found.append(entry)
                    total_chars += len(entry)
                    break  # found this module, move on

        return "\n".join(found) if found else ""

    def _get_readme_context(self, repo_path: Path) -> str:
        """Read the README for model/dataset description."""
        for name in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
            readme = repo_path / name
            if readme.exists():
                try:
                    content = readme.read_text(errors="ignore")
                    if len(content) > 3000:
                        content = content[:3000] + "\n... (truncated)"
                    return content
                except Exception:
                    pass
        return ""

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
