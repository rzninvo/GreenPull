import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { BarChart3, Cpu, Leaf, Loader2, Info } from "lucide-react";
import { submitAnalysis } from "@/api/client";
import InfoButton from "@/components/inform_buttons/InfoButton";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";

const COUNTRIES = [
  { code: "DEU", label: "Germany" },
  { code: "FRA", label: "France" },
  { code: "GBR", label: "United Kingdom" },
  { code: "SWE", label: "Sweden" },
  { code: "NLD", label: "Netherlands" },
  { code: "ESP", label: "Spain" },
  { code: "ITA", label: "Italy" },
  { code: "POL", label: "Poland" },
  { code: "NOR", label: "Norway" },
  { code: "CHE", label: "Switzerland" },
  { code: "USA", label: "United States" },
  { code: "CAN", label: "Canada" },
  { code: "CHN", label: "China" },
  { code: "IND", label: "India" },
  { code: "JPN", label: "Japan" },
  { code: "AUS", label: "Australia" },
  { code: "BRA", label: "Brazil" },
];

const features = [
  {
    icon: BarChart3,
    title: "Measure",
    description: "Run CodeCarbon baseline on your repository to measure exact CO₂ emissions, energy use, and water consumption.",
  },
  {
    icon: Cpu,
    title: "Optimize",
    description: "AI scans your code for opportunities like mixed precision, LoRA, INT8 quantization, and efficient scheduling.",
  },
  {
    icon: Leaf,
    title: "Save",
    description: "Get a ready-to-merge pull request with optimized code and a full sustainability impact report.",
  },
];

const LandingPage = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [patchType, setPatchType] = useState("amp");
  const [country, setCountry] = useState("DEU");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    const githubRegex = /^https?:\/\/github\.com\/[\w.-]+\/[\w.-]+\/?$/;
    if (!githubRegex.test(repoUrl.trim())) {
      setError("Please enter a valid GitHub repository URL");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const result = await submitAnalysis(repoUrl.trim(), patchType, country);
      navigate("/loading", { state: { jobId: result.job_id, repoUrl: repoUrl.trim() } });
    } catch (e: any) {
      setError(e.message || "Failed to submit analysis");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <InfoButton />
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-20">
        <div className="max-w-3xl w-full text-center space-y-6 animate-fade-in">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Leaf className="h-10 w-10 text-green-600" />
            <h1 className="text-5xl font-bold tracking-tight text-foreground">
              Green<span className="text-green-600">Pull</span>
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-lg mx-auto">
            Turn carbon awareness into automated pull requests
          </p>

          <div className="flex flex-col gap-3 max-w-xl mx-auto mt-8">
            <div className="flex flex-col sm:flex-row gap-3">
              <Input
                placeholder="https://github.com/owner/repo"
                value={repoUrl}
                onChange={(e) => { setRepoUrl(e.target.value); setError(""); }}
                onKeyDown={(e) => e.key === "Enter" && !loading && handleAnalyze()}
                className="flex-1 h-12 text-base font-mono"
                disabled={loading}
              />
              <Button
                onClick={handleAnalyze}
                disabled={loading}
                className="h-12 px-8 bg-green-600 hover:bg-green-700 text-white"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                {loading ? "Submitting..." : "Analyze Repository"}
              </Button>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1 space-y-1.5">
                <div className="flex items-center gap-1.5 justify-start">
                  <span className="text-xs font-medium text-foreground">Optimization Strategy</span>
                  <HoverCard openDelay={200} closeDelay={100}>
                    <HoverCardTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                    </HoverCardTrigger>
                    <HoverCardContent className="w-80 text-sm space-y-2" side="top">
                      <p className="font-medium text-foreground">Choose how to optimize your training code</p>
                      <div className="space-y-1.5 text-muted-foreground">
                        <p><strong className="text-foreground">AMP</strong> — Automatic Mixed Precision casts float32 ops to float16, cutting GPU memory by ~50% and training time by ~30%. Best for most workloads.</p>
                        <p><strong className="text-foreground">LoRA</strong> — Low-Rank Adaptation freezes base weights and trains small adapter matrices, reducing trainable parameters by ~90%. Ideal for fine-tuning large models.</p>
                        <p><strong className="text-foreground">Both</strong> — Applies AMP + LoRA together for maximum savings. Recommended if your model supports both.</p>
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                </div>
                <select
                  value={patchType}
                  onChange={(e) => setPatchType(e.target.value)}
                  className="h-10 w-full px-3 rounded-md border border-input bg-background text-sm"
                  disabled={loading}
                >
                  <option value="amp">AMP (Mixed Precision)</option>
                  <option value="lora">LoRA (Parameter-Efficient)</option>
                  <option value="both">Both (AMP + LoRA)</option>
                </select>
                <p className="text-[11px] text-muted-foreground/70 text-left">Affects energy usage & training time</p>
              </div>

              <div className="flex-1 space-y-1.5">
                <div className="flex items-center gap-1.5 justify-start">
                  <span className="text-xs font-medium text-foreground">Training Region</span>
                  <HoverCard openDelay={200} closeDelay={100}>
                    <HoverCardTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                    </HoverCardTrigger>
                    <HoverCardContent className="w-80 text-sm space-y-2" side="top">
                      <p className="font-medium text-foreground">Where will your model be trained?</p>
                      <div className="space-y-1.5 text-muted-foreground">
                        <p>The electrical grid's <strong className="text-foreground">carbon intensity</strong> varies dramatically by country depending on the energy mix (renewables vs fossil fuels).</p>
                        <p>For example, Sweden (~30 gCO₂/kWh, mostly hydro) produces ~10x less carbon per kWh than Poland (~700 gCO₂/kWh, mostly coal).</p>
                        <p>This directly scales your CO₂ emissions estimate. Live data is fetched from the <strong className="text-foreground">Electricity Maps</strong> API when available.</p>
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                </div>
                <select
                  value={country}
                  onChange={(e) => setCountry(e.target.value)}
                  className="h-10 w-full px-3 rounded-md border border-input bg-background text-sm"
                  disabled={loading}
                >
                  {COUNTRIES.map((c) => (
                    <option key={c.code} value={c.code}>{c.label} ({c.code})</option>
                  ))}
                </select>
                <p className="text-[11px] text-muted-foreground/70 text-left">Affects CO₂ emissions estimate</p>
              </div>
            </div>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mt-20 px-4">
          {features.map((f, i) => (
            <Card key={f.title} className="border bg-card hover:shadow-md transition-shadow animate-fade-in" style={{ animationDelay: `${i * 100}ms` }}>
              <CardContent className="p-6 text-center space-y-3">
                <div className="mx-auto w-12 h-12 rounded-full bg-green-50 flex items-center justify-center">
                  <f.icon className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-foreground">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>

      <footer className="py-6 text-center text-sm text-muted-foreground border-t">
        Built with 🌍 for <strong>HackEurope 2026</strong> · Powered by <strong>Crusoe</strong>
      </footer>
    </div>
  );
};

export default LandingPage;
