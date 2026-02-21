import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { BarChart3, Cpu, Leaf } from "lucide-react";

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
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleAnalyze = () => {
    const githubRegex = /^https?:\/\/github\.com\/[\w.-]+\/[\w.-]+\/?$/;
    if (!githubRegex.test(repoUrl.trim())) {
      setError("Please enter a valid GitHub repository URL");
      return;
    }
    setError("");
    navigate("/loading", { state: { repoUrl: repoUrl.trim() } });
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Hero */}
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

          {/* Input */}
          <div className="flex flex-col sm:flex-row gap-3 max-w-xl mx-auto mt-8">
            <Input
              placeholder="https://github.com/owner/repo"
              value={repoUrl}
              onChange={(e) => { setRepoUrl(e.target.value); setError(""); }}
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              className="flex-1 h-12 text-base font-mono"
            />
            <Button onClick={handleAnalyze} className="h-12 px-8 bg-green-600 hover:bg-green-700 text-white">
              Analyze Repository
            </Button>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        {/* Feature cards */}
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

      {/* Footer */}
      <footer className="py-6 text-center text-sm text-muted-foreground border-t">
        Built with 🌍 for <strong>HackEurope 2026</strong> · Powered by <strong>Crusoe</strong>
      </footer>
    </div>
  );
};

export default LandingPage;
