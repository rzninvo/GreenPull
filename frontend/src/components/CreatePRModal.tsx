import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { createPullRequest, type JobResponse } from "@/api/client";
import { GitPullRequest, Loader2, ExternalLink, Leaf, AlertCircle } from "lucide-react";

interface CreatePRModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  job: JobResponse;
}

const CreatePRModal = ({ open, onOpenChange, job }: CreatePRModalProps) => {
  const patchLabel =
    job.patch_type === "amp"
      ? "AMP Mixed Precision"
      : job.patch_type === "lora"
      ? "LoRA Fine-Tuning"
      : "AMP + LoRA";

  const defaultTitle = `GreenPull: Add ${patchLabel} to ${job.detection?.entrypoint_file || "training script"}`;
  const defaultBranch = `greenpull/${job.patch_type || "optimization"}-${job.job_id.slice(0, 8)}`;

  const defaultBody = buildDefaultBody(job, patchLabel);

  const [token, setToken] = useState("");
  const [title, setTitle] = useState(defaultTitle);
  const [body, setBody] = useState(defaultBody);
  const [branchName, setBranchName] = useState(defaultBranch);
  const [baseBranch, setBaseBranch] = useState("main");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [prUrl, setPrUrl] = useState("");

  const handleCreate = async () => {
    if (!token.trim()) {
      setError("GitHub token is required");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const result = await createPullRequest(job.job_id, {
        github_token: token.trim(),
        title,
        body,
        branch_name: branchName,
        base_branch: baseBranch,
      });
      setPrUrl(result.pr_url);
    } catch (e: any) {
      setError(e.message || "Failed to create pull request");
    } finally {
      setLoading(false);
    }
  };

  // Success state
  if (prUrl) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-lg">
          <div className="text-center space-y-4 py-4">
            <div className="mx-auto w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
              <GitPullRequest className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Pull Request Created!</h3>
              <p className="text-sm text-muted-foreground mt-1">Your optimization has been submitted.</p>
            </div>
            <a
              href={prUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-green-600 hover:bg-green-700 text-white text-sm font-medium transition-colors"
            >
              <ExternalLink className="h-4 w-4" />
              View on GitHub
            </a>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-green-600" />
            Create Pull Request
          </DialogTitle>
          <DialogDescription>
            Push the optimization to <span className="font-mono text-foreground">{job.repo_url.replace("https://github.com/", "")}</span>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 mt-2">
          {/* GitHub Token */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-foreground">GitHub Token</label>
            <Input
              type="password"
              placeholder="ghp_xxxxxxxxxxxx"
              value={token}
              onChange={(e) => { setToken(e.target.value); setError(""); }}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Personal access token with <span className="font-mono">repo</span> scope
            </p>
          </div>

          {/* Title */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-foreground">PR Title</label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="text-sm"
            />
          </div>

          {/* Branch names */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-foreground">Branch</label>
              <Input
                value={branchName}
                onChange={(e) => setBranchName(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-foreground">Base</label>
              <Input
                value={baseBranch}
                onChange={(e) => setBaseBranch(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          {/* Description */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-foreground">Description</label>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows={6}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono resize-y"
            />
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 text-sm text-red-600 bg-red-50 rounded-md p-3">
              <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={loading}
              className="bg-green-600 hover:bg-green-700 text-white gap-2"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitPullRequest className="h-4 w-4" />}
              {loading ? "Creating..." : "Create Pull Request"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

function buildDefaultBody(job: JobResponse, patchLabel: string): string {
  const lines: string[] = [];
  lines.push(`## GreenPull Optimization\n`);
  lines.push(`**File:** \`${job.detection?.entrypoint_file || "unknown"}\``);
  lines.push(`**Technique:** ${patchLabel}\n`);

  if (job.savings) {
    lines.push(`### Estimated Savings`);
    lines.push(`- **CO2 reduced:** ${job.savings.emissions_saved_kg.toFixed(4)} kg (${job.savings.emissions_saved_pct.toFixed(1)}%)`);
    lines.push(`- **Energy saved:** ${job.savings.energy_saved_kwh.toFixed(4)} kWh (${job.savings.energy_saved_pct.toFixed(1)}%)`);
  }

  if (job.comparisons) {
    lines.push(`\n### Real-World Equivalents`);
    if (job.comparisons.tree_months) lines.push(`- ${job.comparisons.tree_months.toFixed(1)} tree-months of CO2 absorption`);
    if (job.comparisons.car_km) lines.push(`- ${job.comparisons.car_km.toFixed(1)} km of driving avoided`);
    if (job.comparisons.smartphone_charges) lines.push(`- ${Math.round(job.comparisons.smartphone_charges)} smartphone charges saved`);
  }

  lines.push(`\n---\n*Generated by [GreenPull](https://github.com) — carbon-aware ML optimization*`);
  return lines.join("\n");
}

export default CreatePRModal;
