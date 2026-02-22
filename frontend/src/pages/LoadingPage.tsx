import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { getJobStatus, type JobResponse } from "@/api/client";
import { Check, Loader2, Circle, Leaf, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

type StepStatus = "pending" | "in_progress" | "done";

const STEPS = [
  { label: "Cloning repository", statuses: ["cloning"] },
  { label: "Detecting training entrypoint", statuses: ["analyzing"] },
  { label: "Extracting training configuration", statuses: ["extracting_config"] },
  { label: "Estimating baseline emissions", statuses: ["estimating_baseline"] },
  { label: "Generating optimization patch", statuses: ["patching"] },
  { label: "Estimating optimized emissions", statuses: ["estimating_optimized"] },
];

function getStepStatuses(backendStatus: string): StepStatus[] {
  const statusOrder = [
    "cloning", "analyzing", "extracting_config",
    "estimating_baseline", "patching", "estimating_optimized",
  ];
  const currentIdx = statusOrder.indexOf(backendStatus);
  if (currentIdx === -1) return STEPS.map(() => "pending");
  return STEPS.map((_, i) => {
    if (i < currentIdx) return "done";
    if (i === currentIdx) return "in_progress";
    return "pending";
  });
}

const LoadingPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as { jobId?: string; repoUrl?: string } | null;
  const jobId = state?.jobId;
  const repoUrl = state?.repoUrl || "";

  const [statuses, setStatuses] = useState<StepStatus[]>(STEPS.map(() => "pending"));
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId) {
      navigate("/");
      return;
    }

    const poll = async () => {
      try {
        const job: JobResponse = await getJobStatus(jobId);

        if (job.status === "completed") {
          setStatuses(STEPS.map(() => "done"));
          setTimeout(() => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            navigate("/results", { state: { job, repoUrl } });
          }, 600);
          return;
        }

        if (job.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setError(job.error_message || "Analysis failed");
          return;
        }

        setStatuses(getStepStatuses(job.status));
      } catch (e: any) {
        console.error("Poll error:", e);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId, navigate, repoUrl]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background px-4">
      <div className="max-w-md w-full space-y-8 animate-fade-in">
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Leaf className="h-7 w-7 text-green-600" />
            <h1 className="text-2xl font-bold text-foreground">Green<span className="text-green-600">Pull</span></h1>
          </div>
          <p className="text-sm text-muted-foreground font-mono truncate">{repoUrl}</p>
        </div>

        <div className="space-y-1">
          {STEPS.map((step, i) => (
            <div
              key={i}
              className={`flex items-center gap-3 py-3 px-4 rounded-lg transition-all duration-300 ${
                statuses[i] === "in_progress"
                  ? "bg-blue-50 text-blue-700"
                  : statuses[i] === "done"
                  ? "text-green-700"
                  : "text-muted-foreground"
              }`}
            >
              <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center">
                {statuses[i] === "done" ? (
                  <Check className="h-5 w-5 text-green-600" />
                ) : statuses[i] === "in_progress" ? (
                  <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                ) : (
                  <Circle className="h-4 w-4 text-muted-foreground/40" />
                )}
              </div>
              <span className={`text-sm ${statuses[i] === "in_progress" ? "font-medium" : ""}`}>
                {step.label}
              </span>
            </div>
          ))}
        </div>

        {error && (
          <div className="flex items-start gap-3 p-4 rounded-lg bg-red-50 text-red-700 text-sm">
            <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Analysis failed</p>
              <p className="mt-1 text-xs font-mono whitespace-pre-wrap">{error}</p>
            </div>
          </div>
        )}

        <Button
          variant="outline"
          onClick={() => navigate("/")}
          className="mt-6 gap-2 text-muted-foreground"
        >
          <Circle className="h-3 w-3 fill-red-500 text-red-500" />
          Stop Analysis
        </Button>
      </div>
    </div>
  );
};

export default LoadingPage;
