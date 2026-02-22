import type { DiffLine, FileDiff } from "@/data/mockData";

// --- Types matching backend schemas ---

export interface AnalyzeRequest {
  repo_url: string;
  patch_type: "amp" | "lora" | "both";
  country_iso_code: string;
}

export interface JobCreatedResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface EmissionsDetail {
  emissions_kg: number | null;
  energy_kwh: number | null;
  duration_s: number | null;
  power_w: number | null;
  cpu_energy_kwh: number | null;
  gpu_energy_kwh: number | null;
  memory_energy_kwh: number | null;
  cpu_model: string | null;
  gpu_model: string | null;
}

export interface DetectionResult {
  entrypoint_file: string | null;
  run_command: string | null;
  framework: string | null;
  reasoning: string | null;
}

export interface TrainingConfig {
  model_type: string | null;
  model_name: string | null;
  parameter_count_millions: number | null;
  framework: string | null;
  epochs: number | null;
  batch_size: number | null;
  dataset_size_estimate: string | null;
  estimated_runtime_hours: number | null;
  gpu_type: string | null;
  num_gpus: number | null;
  reasoning: string | null;
}

export interface Comparisons {
  tree_months: number | null;
  car_km: number | null;
  smartphone_charges: number | null;
  streaming_hours: number | null;
  flight_fraction: number | null;
  led_bulb_hours: number | null;
}

export interface Savings {
  emissions_saved_kg: number;
  emissions_saved_pct: number;
  energy_saved_kwh: number;
  energy_saved_pct: number;
}

export interface JobResponse {
  job_id: string;
  repo_url: string;
  status: string;
  created_at: string;
  estimation_method: string | null;
  detection: DetectionResult | null;
  training_config: TrainingConfig | null;
  baseline: EmissionsDetail | null;
  optimized: EmissionsDetail | null;
  patch_type: string | null;
  patch_diff: string | null;
  savings: Savings | null;
  comparisons: Comparisons | null;
  error_message: string | null;
}

// --- API calls ---

export async function submitAnalysis(
  repoUrl: string,
  patchType: string,
  countryCode: string
): Promise<JobCreatedResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_url: repoUrl,
      patch_type: patchType,
      country_iso_code: countryCode,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

// --- Pull Request ---

export interface CreatePRRequest {
  github_token: string;
  title: string;
  body: string;
  branch_name: string;
  base_branch: string;
}

export interface CreatePRResponse {
  pr_number: number;
  pr_url: string;
}

export async function createPullRequest(
  jobId: string,
  req: CreatePRRequest
): Promise<CreatePRResponse> {
  const res = await fetch(`/api/jobs/${jobId}/create-pr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

// --- Unified diff parser ---

export function parseUnifiedDiff(
  diffStr: string,
  filename: string,
  patchType: string
): FileDiff {
  const lines = diffStr.split("\n");
  const diffLines: DiffLine[] = [];
  let oldLine = 0;
  let newLine = 0;

  for (const line of lines) {
    // Skip diff headers
    if (
      line.startsWith("---") ||
      line.startsWith("+++") ||
      line.startsWith("diff ")
    ) {
      continue;
    }

    // Hunk header: @@ -oldStart,count +newStart,count @@
    const hunkMatch = line.match(/^@@ -(\d+),?\d* \+(\d+),?\d* @@/);
    if (hunkMatch) {
      const nextOld = parseInt(hunkMatch[1]);
      const nextNew = parseInt(hunkMatch[2]);
      // Add hidden lines indicator if there's a gap
      if (diffLines.length > 0 && (nextOld > oldLine + 1 || nextNew > newLine + 1)) {
        const hiddenCount = Math.max(nextOld - oldLine - 1, nextNew - newLine - 1);
        if (hiddenCount > 0) {
          diffLines.push({ type: "hidden", content: "", hiddenCount });
        }
      }
      oldLine = nextOld;
      newLine = nextNew;
      continue;
    }

    if (line.startsWith("-")) {
      diffLines.push({
        type: "removed",
        content: line.slice(1),
        oldLineNum: oldLine++,
      });
    } else if (line.startsWith("+")) {
      diffLines.push({
        type: "added",
        content: line.slice(1),
        newLineNum: newLine++,
      });
    } else if (line.startsWith(" ") || line === "") {
      diffLines.push({
        type: "unchanged",
        content: line.startsWith(" ") ? line.slice(1) : line,
        oldLineNum: oldLine++,
        newLineNum: newLine++,
      });
    }
  }

  const patchLabel =
    patchType === "amp"
      ? "Mixed Precision (AMP)"
      : patchType === "lora"
      ? "LoRA Fine-Tuning"
      : "AMP + LoRA";

  return {
    filename,
    optimization: patchLabel,
    co2Savings: "",
    lines: diffLines,
  };
}
