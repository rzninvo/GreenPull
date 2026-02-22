import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { mockMetrics, regionData, greenWindow } from "@/data/mockData";
import CodeDiff from "@/components/CodeDiff";
import { parseUnifiedDiff, type JobResponse } from "@/api/client";
import {
  Leaf, GitPullRequest, Cloud, Zap, Droplets, Gauge, MapPin, Clock, ArrowRight,
  TreePine, Car, Smartphone, Play, Plane, Lightbulb, Cpu, BrainCircuit,
} from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

const chartConfig = {
  Before: { label: "Before", color: "hsl(0, 72%, 65%)" },
  After: { label: "After", color: "hsl(142, 76%, 36%)" },
};

const fmt = (v: number | null | undefined, decimals = 2) => {
  if (v == null) return "—";
  if (Math.abs(v) < 0.01) return v.toExponential(1);
  return v.toFixed(decimals);
};

const ResultsPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as { job?: JobResponse; repoUrl?: string } | null;
  const job = state?.job;

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <p className="text-muted-foreground">No results data. Run an analysis first.</p>
          <Button onClick={() => navigate("/")}>Go Home</Button>
        </div>
      </div>
    );
  }

  const { baseline, optimized, savings, comparisons, training_config, detection, patch_diff, patch_type } = job;

  // Build metrics for cards
  const realMetrics = [
    {
      key: "co2",
      label: "CO₂ Emissions",
      unit: "kg",
      before: baseline?.emissions_kg,
      after: optimized?.emissions_kg,
      reduction: savings?.emissions_saved_pct,
      icon: Cloud,
    },
    {
      key: "energy",
      label: "Energy Usage",
      unit: "kWh",
      before: baseline?.energy_kwh,
      after: optimized?.energy_kwh,
      reduction: savings?.energy_saved_pct,
      icon: Zap,
    },
  ];

  const mockCards = Object.entries(mockMetrics).map(([key, m]) => ({
    key,
    label: m.label,
    unit: m.unit,
    before: m.before,
    after: m.after,
    reduction: Math.round(((m.before - m.after) / m.before) * 100),
    icon: key === "water" ? Droplets : Gauge,
  }));

  const allMetrics = [...realMetrics, ...mockCards];

  // Bar chart data
  const barData = allMetrics.map((m) => ({
    name: m.label.split(" ")[0],
    Before: m.before ?? 0,
    After: m.after ?? 0,
  }));

  // Pie chart data from baseline energy breakdown
  const cpuE = baseline?.cpu_energy_kwh ?? 0;
  const gpuE = baseline?.gpu_energy_kwh ?? 0;
  const memE = Math.max((baseline?.energy_kwh ?? 0) - cpuE - gpuE, 0);
  const pieData = [
    { name: "GPU", value: gpuE, color: "hsl(142, 76%, 36%)" },
    { name: "CPU", value: cpuE, color: "hsl(199, 89%, 48%)" },
    { name: "Memory", value: memE, color: "hsl(262, 83%, 58%)" },
  ].filter((d) => d.value > 0);

  // Parse diff
  const diffs = patch_diff && detection?.entrypoint_file
    ? [parseUnifiedDiff(patch_diff, detection.entrypoint_file, patch_type || "amp")]
    : [];
  if (diffs.length > 0 && savings) {
    diffs[0].co2Savings = `Saves ~${fmt(savings.emissions_saved_kg)} kg CO₂`;
  }

  // Comparisons list
  const comparisonItems = comparisons ? [
    { icon: TreePine, label: "Tree-months of CO₂ absorption", value: fmt(comparisons.tree_months, 1) },
    { icon: Car, label: "km of driving avoided", value: fmt(comparisons.car_km, 1) },
    { icon: Smartphone, label: "smartphone charges saved", value: fmt(comparisons.smartphone_charges, 0) },
    { icon: Play, label: "hours of video streaming", value: fmt(comparisons.streaming_hours, 1) },
    { icon: Plane, label: "of a Paris–NYC flight", value: `${fmt((comparisons.flight_fraction ?? 0) * 100, 2)}%` },
    { icon: Lightbulb, label: "hours of LED bulb", value: fmt(comparisons.led_bulb_hours, 1) },
  ] : [];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Leaf className="h-6 w-6 text-green-600" />
            <span className="text-xl font-bold text-foreground">Green<span className="text-green-600">Pull</span></span>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={() => navigate("/")} className="gap-2">
              <ArrowRight className="h-4 w-4 rotate-180" />
              New Analysis
            </Button>
            <Button className="bg-green-600 hover:bg-green-700 text-white gap-2">
              <GitPullRequest className="h-4 w-4" />
              Create Pull Request
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-10">
        {/* Training config summary */}
        {training_config && (
          <section>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center gap-2 mb-3">
                  <BrainCircuit className="h-5 w-5 text-green-600" />
                  <h2 className="text-base font-semibold text-foreground">Detected Training Configuration</h2>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-sm">
                  {training_config.model_name && (
                    <div><span className="text-muted-foreground">Model</span><p className="font-medium font-mono">{training_config.model_name}</p></div>
                  )}
                  {training_config.parameter_count_millions && (
                    <div><span className="text-muted-foreground">Parameters</span><p className="font-medium font-mono">{training_config.parameter_count_millions}M</p></div>
                  )}
                  {training_config.epochs && (
                    <div><span className="text-muted-foreground">Epochs</span><p className="font-medium font-mono">{training_config.epochs}</p></div>
                  )}
                  {training_config.gpu_type && (
                    <div><span className="text-muted-foreground">GPU</span><p className="font-medium font-mono">{training_config.gpu_type} x{training_config.num_gpus ?? 1}</p></div>
                  )}
                  {training_config.estimated_runtime_hours && (
                    <div><span className="text-muted-foreground">Est. Runtime</span><p className="font-medium font-mono">{fmt(training_config.estimated_runtime_hours, 1)}h</p></div>
                  )}
                  {detection?.framework && (
                    <div><span className="text-muted-foreground">Framework</span><p className="font-medium font-mono">{detection.framework}</p></div>
                  )}
                </div>
              </CardContent>
            </Card>
          </section>
        )}

        {/* Metrics row */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {allMetrics.map((m) => {
            const Icon = m.icon;
            const reduction = m.reduction != null ? Math.round(m.reduction) : 0;
            return (
              <Card key={m.key} className="border">
                <CardContent className="p-5 space-y-2">
                  <div className="flex items-center gap-2 text-muted-foreground text-sm">
                    <Icon className="h-4 w-4" />
                    {m.label}
                  </div>
                  <div className="flex items-baseline gap-3">
                    <span className="text-2xl font-bold text-foreground">
                      {fmt(m.after)} <span className="text-sm font-normal text-muted-foreground">{m.unit}</span>
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground line-through">{fmt(m.before)} {m.unit}</span>
                    {reduction > 0 && (
                      <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0 text-xs">
                        ↓ {reduction}%
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </section>

        {/* Real-world comparisons */}
        {comparisonItems.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold text-foreground mb-4">Savings Equivalent To</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              {comparisonItems.map((item) => (
                <Card key={item.label} className="border">
                  <CardContent className="p-4 text-center space-y-2">
                    <item.icon className="h-6 w-6 text-green-600 mx-auto" />
                    <p className="text-xl font-bold text-foreground">{item.value}</p>
                    <p className="text-xs text-muted-foreground leading-tight">{item.label}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        )}

        {/* Code diffs */}
        {diffs.length > 0 && (
          <section className="space-y-4">
            <h2 className="text-lg font-semibold text-foreground">Optimization Diff</h2>
            {diffs.map((diff, i) => (
              <CodeDiff key={i} diff={diff} />
            ))}
          </section>
        )}

        {/* Charts */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Before vs After</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[260px] w-full">
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="name" className="text-xs" />
                  <YAxis className="text-xs" />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="Before" fill="hsl(0, 72%, 65%)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="After" fill="hsl(142, 76%, 36%)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          {pieData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Energy Breakdown (Baseline)</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center">
                <div className="h-[260px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={4}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value.toFixed(4)} kWh`}
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Region & scheduling (mock) */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <MapPin className="h-4 w-4" /> Region Recommendation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex-1 p-3 rounded-lg bg-red-50 text-center">
                  <p className="text-xs text-muted-foreground">Current</p>
                  <p className="font-semibold text-sm text-foreground">{regionData.current.name}</p>
                  <p className="text-red-600 font-mono text-lg font-bold">{regionData.current.intensity}</p>
                  <p className="text-xs text-muted-foreground">{regionData.current.unit}</p>
                </div>
                <ArrowRight className="h-5 w-5 text-green-600 flex-shrink-0" />
                <div className="flex-1 p-3 rounded-lg bg-green-50 text-center">
                  <p className="text-xs text-muted-foreground">Recommended</p>
                  <p className="font-semibold text-sm text-foreground">{regionData.recommended.name}</p>
                  <p className="text-green-600 font-mono text-lg font-bold">{regionData.recommended.intensity}</p>
                  <p className="text-xs text-muted-foreground">{regionData.recommended.unit}</p>
                </div>
              </div>
              <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0">
                ↓ {regionData.savingsPercent}% carbon intensity
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="h-4 w-4" /> Green Window (72h Forecast)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 rounded-lg bg-green-50 space-y-1">
                <p className="text-sm font-medium text-foreground">Best time to run</p>
                <p className="font-mono text-green-700 font-semibold">{greenWindow.bestTime}</p>
              </div>
              <div className="flex gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Grid now: </span>
                  <span className="font-mono font-semibold text-foreground">{greenWindow.normalIntensity} gCO₂/kWh</span>
                </div>
                <div>
                  <span className="text-muted-foreground">At best: </span>
                  <span className="font-mono font-semibold text-green-700">{greenWindow.gridIntensity} gCO₂/kWh</span>
                </div>
              </div>
              <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0">
                Additional savings: {greenWindow.additionalSavings}
              </Badge>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
};

export default ResultsPage;
