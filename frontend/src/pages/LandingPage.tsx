import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { BarChart3, Cpu, Leaf, Loader2 } from "lucide-react";
import { submitAnalysis } from "@/api/client";

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
    title: "Estimate",
    description: "Analyze your ML training code and estimate CO₂ emissions using Green Algorithms formulas — no code execution needed.",
  },
  {
    icon: Cpu,
    title: "Optimize",
    description: "AI generates AMP mixed precision and LoRA patches to reduce energy consumption and training time.",
  },
  {
    icon: Leaf,
    title: "Save",
    description: "See before/after emissions, energy savings, and real-world comparisons like tree-months and car km avoided.",
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

            <div className="flex flex-col sm:flex-row gap-3">
              <select
                value={patchType}
                onChange={(e) => setPatchType(e.target.value)}
                className="h-10 px-3 rounded-md border border-input bg-background text-sm flex-1"
                disabled={loading}
              >
                <option value="amp">AMP (Mixed Precision)</option>
                <option value="lora">LoRA (Parameter-Efficient)</option>
                <option value="both">Both (AMP + LoRA)</option>
              </select>
              <select
                value={country}
                onChange={(e) => setCountry(e.target.value)}
                className="h-10 px-3 rounded-md border border-input bg-background text-sm flex-1"
                disabled={loading}
              >
                {COUNTRIES.map((c) => (
                  <option key={c.code} value={c.code}>{c.label} ({c.code})</option>
                ))}
              </select>
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
