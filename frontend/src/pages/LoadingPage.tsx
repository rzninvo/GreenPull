import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { pipelineSteps } from "@/data/mockData";
import { Check, Loader2, Circle, Leaf } from "lucide-react";
import { Button } from "@/components/ui/button";

type StepStatus = "pending" | "in_progress" | "done";

const LoadingPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const repoUrl = (location.state as any)?.repoUrl || "https://github.com/example/repo";
  const [statuses, setStatuses] = useState<StepStatus[]>(pipelineSteps.map(() => "pending"));

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;
    let currentStep = 0;

    const advance = () => {
      if (currentStep >= pipelineSteps.length) {
        // All done — navigate after brief pause
        timeout = setTimeout(() => navigate("/results", { state: { repoUrl } }), 600);
        return;
      }
      // Set current step to in_progress
      setStatuses((prev) => prev.map((s, i) => (i === currentStep ? "in_progress" : s)));

      // After duration, mark done and advance
      timeout = setTimeout(() => {
        setStatuses((prev) => prev.map((s, i) => (i === currentStep ? "done" : s)));
        currentStep++;
        setTimeout(advance, 300);
      }, pipelineSteps[currentStep].duration);
    };

    advance();
    return () => clearTimeout(timeout);
  }, [navigate, repoUrl]);

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
          {pipelineSteps.map((step, i) => (
            <div
              key={step.id}
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
