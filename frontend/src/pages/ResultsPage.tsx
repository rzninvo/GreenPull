import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { mockMetrics, regionData, greenWindow } from "@/data/mockData";
import CodeDiff from "@/components/CodeDiff";
import CreatePRModal from "@/components/CreatePRModal";
import { parseMultiFileDiff, type JobResponse } from "@/api/client";
import {
  Leaf, GitPullRequest, Cloud, Zap, Droplets, Gauge, MapPin, Clock, ArrowRight,
  TreePine, Car, Smartphone, Play, Plane, Lightbulb, Cpu, BrainCircuit, Info,
} from "lucide-react";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";
import { useNavigate, useLocation } from "react-router-dom";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
// @ts-ignore – react-katex has no type declarations
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const chartConfig = {
  Before: { label: "Before", color: "hsl(0, 72%, 65%)" },
  After: { label: "After", color: "hsl(142, 76%, 36%)" },
};

const fmt = (v: number | null | undefined, decimals = 2) => {
  if (v == null) return "—";
  if (Math.abs(v) < 0.01) return v.toExponential(1);
  return v.toFixed(decimals);
};

const METRIC_INFO: Record<string, { description: string; formula?: string; variables?: Array<{ sym: string; desc: string }> }> = {
  co2: {
    description: "Total carbon dioxide equivalent emitted during the training run. Estimated from energy consumption and grid carbon intensity for your region.",
    formula: "\\text{CO}_2 = E \\times I",
    variables: [
      { sym: "E", desc: "Energy consumed (kWh)" },
      { sym: "I", desc: "Carbon intensity of the grid (kgCO₂/kWh)" },
    ],
  },
  energy: {
    description: "Total electrical energy consumed during model training, including GPU, CPU, and memory subsystems. Measured or estimated via hardware TDP and runtime.",
    formula: "E = (P_{\\text{gpu}} + P_{\\text{cpu}} + P_{\\text{mem}}) \\times t \\times \\text{PUE}",
    variables: [
      { sym: "P", desc: "Power draw per component (W)" },
      { sym: "t", desc: "Training duration (h)" },
      { sym: "\\text{PUE}", desc: "Power Usage Effectiveness (~1.1)" },
    ],
  },
  water: {
    description: "Estimated water consumed for datacenter cooling during the training run. Datacenters use evaporative cooling that consumes fresh water.",
    formula: "\\text{Water} = E \\times \\text{WUE}",
    variables: [
      { sym: "E", desc: "Energy consumed (kWh)" },
      { sym: "\\text{WUE}", desc: "Water Usage Effectiveness (L/kWh, typically 1.8)" },
    ],
  },
  sci: {
    description: "Software Carbon Intensity — a Green Software Foundation (GSF) standard metric that quantifies the carbon cost per functional unit of software.",
    formula: "\\text{SCI} = \\frac{(E \\times I) + M}{R}",
    variables: [
      { sym: "E", desc: "Energy consumed (kWh)" },
      { sym: "I", desc: "Carbon intensity (gCO₂/kWh)" },
      { sym: "M", desc: "Embodied emissions (gCO₂)" },
      { sym: "R", desc: "Functional unit (1 training run)" },
    ],
  },
};

const MetricTooltip = ({ metricKey }: { metricKey: string }) => {
  const info = METRIC_INFO[metricKey];
  if (!info) return null;
  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
      </HoverCardTrigger>
      <HoverCardContent className="w-80 text-sm space-y-2" side="top">
        <p className="text-muted-foreground leading-snug">{info.description}</p>
        {info.formula && (
          <div className="rounded-md bg-muted/50 px-3 py-2.5 space-y-2">
            <div className="flex justify-center">
              <BlockMath math={info.formula} />
            </div>
            {info.variables && (
              <div className="text-xs text-muted-foreground space-y-0.5 pt-1 border-t border-border/50">
                {info.variables.map((v) => (
                  <div key={v.sym} className="flex items-baseline gap-1.5">
                    <InlineMath math={v.sym} />
                    <span>= {v.desc}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </HoverCardContent>
    </HoverCard>
  );
};

const ResultsPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as { job?: JobResponse; repoUrl?: string } | null;
  const job = state?.job;
  const [prModalOpen, setPrModalOpen] = useState(false);

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

  // Water usage: use real data if available, else mock
  const waterData = job.water_usage;
  const extraCards = [];
  if (waterData && waterData.baseline_liters != null) {
    const wBefore = waterData.baseline_liters;
    const wAfter = waterData.optimized_liters ?? wBefore;
    extraCards.push({
      key: "water",
      label: "Water Usage",
      unit: "liters",
      before: wBefore,
      after: wAfter,
      reduction: wBefore > 0 ? Math.round(((wBefore - wAfter) / wBefore) * 100) : 0,
      icon: Droplets,
    });
  } else {
    extraCards.push({
      key: "water",
      label: mockMetrics.water.label,
      unit: mockMetrics.water.unit,
      before: mockMetrics.water.before,
      after: mockMetrics.water.after,
      reduction: Math.round(((mockMetrics.water.before - mockMetrics.water.after) / mockMetrics.water.before) * 100),
      icon: Droplets,
    });
  }
  // SCI score: SCI = ((E × I) + M) / R
  // E = energy (kWh), I = carbon intensity (gCO2/kWh), M = embodied emissions, R = 1 run
  // Embodied estimate: ~150 kg CO2 per GPU manufactured, ~35040h lifespan → ~4.28 gCO2/h/GPU
  {
    const ciValue = job.carbon_intensity_info?.value ?? 400; // gCO2/kWh fallback
    const numGpus = training_config?.num_gpus ?? 1;
    const runtimeH = training_config?.estimated_runtime_hours ?? 1;
    const embodiedPerGpuPerH = 4.28; // gCO2/h (150kg / 35040h)
    const embodied = embodiedPerGpuPerH * numGpus * runtimeH; // gCO2

    const sciBefore = baseline?.energy_kwh != null
      ? (baseline.energy_kwh * ciValue + embodied)
      : null;
    const sciAfter = optimized?.energy_kwh != null
      ? (optimized.energy_kwh * ciValue + embodied)
      : null;

    if (sciBefore != null && sciAfter != null) {
      extraCards.push({
        key: "sci",
        label: "SCI Score",
        unit: "gCO₂eq/run",
        before: Math.round(sciBefore * 100) / 100,
        after: Math.round(sciAfter * 100) / 100,
        reduction: sciBefore > 0 ? Math.round(((sciBefore - sciAfter) / sciBefore) * 100) : 0,
        icon: Gauge,
      });
    } else {
      extraCards.push({
        key: "sci",
        label: mockMetrics.sci.label,
        unit: mockMetrics.sci.unit,
        before: mockMetrics.sci.before,
        after: mockMetrics.sci.after,
        reduction: Math.round(((mockMetrics.sci.before - mockMetrics.sci.after) / mockMetrics.sci.before) * 100),
        icon: Gauge,
      });
    }
  }

  const allMetrics = [...realMetrics, ...extraCards];

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

  // Parse diff (multi-file aware)
  const diffs = patch_diff
    ? parseMultiFileDiff(patch_diff, patch_type || "amp", job.patched_files ?? undefined)
    : [];
  if (diffs.length > 0 && savings) {
    diffs[0].co2Savings = `Saves ~${fmt(savings.emissions_saved_kg)} kg CO₂`;
  }

  // Comparisons list
  const comparisonItems = comparisons ? [
    { icon: TreePine, label: "Tree-months of CO₂ absorption", value: fmt(comparisons.tree_months, 1), tip: "A mature tree absorbs ~21 kg CO₂/year. This shows how many months a single tree would need to offset the saved emissions." },
    { icon: Car, label: "km of driving avoided", value: fmt(comparisons.car_km, 1), tip: "An average car emits ~0.21 kg CO₂/km. This converts the CO₂ savings to equivalent driving distance." },
    { icon: Smartphone, label: "smartphone charges saved", value: fmt(comparisons.smartphone_charges, 0), tip: "Charging a smartphone uses ~0.012 kWh. This shows how many full charges the saved energy could power." },
    { icon: Play, label: "hours of video streaming", value: fmt(comparisons.streaming_hours, 1), tip: "Streaming video uses ~0.036 kWh per hour. This converts saved energy to streaming time." },
    { icon: Plane, label: "of a Paris–NYC flight", value: `${fmt((comparisons.flight_fraction ?? 0) * 100, 2)}%`, tip: "A Paris–NYC economy flight emits ~350 kg CO₂ per passenger. This shows the CO₂ savings as a fraction of that flight." },
    { icon: Lightbulb, label: "hours of LED bulb", value: fmt(comparisons.led_bulb_hours, 1), tip: "A 10W LED bulb uses 0.01 kWh per hour. This shows how long an LED could run on the saved energy." },
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
            <Button
              onClick={() => setPrModalOpen(true)}
              className="bg-green-600 hover:bg-green-700 text-white gap-2"
            >
              <GitPullRequest className="h-4 w-4" />
              Create Pull Request
            </Button>
          </div>
        </div>
      </header>

      <CreatePRModal open={prModalOpen} onOpenChange={setPrModalOpen} job={job} />

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
                    <MetricTooltip metricKey={m.key} />
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
                <HoverCard key={item.label} openDelay={200} closeDelay={100}>
                  <HoverCardTrigger asChild>
                    <Card className="border cursor-help">
                      <CardContent className="p-4 text-center space-y-2">
                        <item.icon className="h-6 w-6 text-green-600 mx-auto" />
                        <p className="text-xl font-bold text-foreground">{item.value}</p>
                        <p className="text-xs text-muted-foreground leading-tight">{item.label}</p>
                      </CardContent>
                    </Card>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-64 text-sm" side="top">
                    <p className="text-muted-foreground leading-snug">{item.tip}</p>
                  </HoverCardContent>
                </HoverCard>
              ))}
            </div>
          </section>
        )}

        {/* Code diffs */}
        {diffs.length > 0 && (
          <section className="space-y-4">
            <h2 className="text-lg font-semibold text-foreground">
              Optimization Diff
              {diffs.length > 1 && (
                <span className="text-sm font-normal text-muted-foreground ml-2">
                  ({diffs.length} files)
                </span>
              )}
            </h2>
            {diffs.map((diff, i) => (
              <CodeDiff key={i} diff={diff} />
            ))}
          </section>
        )}

        {/* Charts */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                Before vs After
                <HoverCard openDelay={200} closeDelay={100}>
                  <HoverCardTrigger asChild>
                    <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                  </HoverCardTrigger>
                  <HoverCardContent className="w-64 text-sm" side="top">
                    <p className="text-muted-foreground leading-snug">Side-by-side comparison of each environmental metric before and after applying the optimization patches.</p>
                  </HoverCardContent>
                </HoverCard>
              </CardTitle>
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
                <CardTitle className="text-base flex items-center gap-2">
                  Energy Breakdown (Baseline)
                  <HoverCard openDelay={200} closeDelay={100}>
                    <HoverCardTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                    </HoverCardTrigger>
                    <HoverCardContent className="w-64 text-sm" side="top">
                      <p className="text-muted-foreground leading-snug">How the baseline energy consumption is distributed across GPU, CPU, and memory. GPU-intensive workloads benefit most from mixed precision (AMP).</p>
                    </HoverCardContent>
                  </HoverCard>
                </CardTitle>
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

        {/* Region & scheduling */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <MapPin className="h-4 w-4" /> Region Recommendation
                <HoverCard openDelay={200} closeDelay={100}>
                  <HoverCardTrigger asChild>
                    <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                  </HoverCardTrigger>
                  <HoverCardContent className="w-72 text-sm" side="top">
                    <p className="text-muted-foreground leading-snug">Suggests a cloud region with lower carbon intensity. Grid carbon intensity varies by country based on the energy mix (renewables vs fossil fuels). Data sourced from Electricity Maps.</p>
                  </HoverCardContent>
                </HoverCard>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {(() => {
                const rr = job.region_recommendation;
                const currentName = rr?.current_zone || regionData.current.name;
                const currentIntensity = rr?.current_intensity ?? regionData.current.intensity;
                const recName = rr?.recommended_region_name || regionData.recommended.name;
                const recIntensity = rr?.recommended_intensity ?? regionData.recommended.intensity;
                const savingsPct = rr?.savings_pct ?? regionData.savingsPercent;
                return (
                  <>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 p-3 rounded-lg bg-red-50 text-center">
                        <p className="text-xs text-muted-foreground">Current</p>
                        <p className="font-semibold text-sm text-foreground">{currentName}</p>
                        <p className="text-red-600 font-mono text-lg font-bold">{fmt(currentIntensity, 0)}</p>
                        <p className="text-xs text-muted-foreground">gCO₂/kWh</p>
                      </div>
                      <ArrowRight className="h-5 w-5 text-green-600 flex-shrink-0" />
                      <div className="flex-1 p-3 rounded-lg bg-green-50 text-center">
                        <p className="text-xs text-muted-foreground">Recommended</p>
                        <p className="font-semibold text-sm text-foreground">{recName}</p>
                        <p className="text-green-600 font-mono text-lg font-bold">{fmt(recIntensity, 0)}</p>
                        <p className="text-xs text-muted-foreground">gCO₂/kWh</p>
                      </div>
                    </div>
                    <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0">
                      ↓ {fmt(savingsPct, 0)}% carbon intensity
                    </Badge>
                  </>
                );
              })()}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="h-4 w-4" /> Green Window (72h Forecast)
                <HoverCard openDelay={200} closeDelay={100}>
                  <HoverCardTrigger asChild>
                    <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                  </HoverCardTrigger>
                  <HoverCardContent className="w-72 text-sm" side="top">
                    <p className="text-muted-foreground leading-snug">Identifies the lowest-carbon time window in the next 72 hours to run your training job. Grid carbon intensity fluctuates with renewable energy availability (solar, wind).</p>
                  </HoverCardContent>
                </HoverCard>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {(() => {
                const gw = job.green_window;
                const bestTime = gw?.best_window_start
                  ? `${new Date(gw.best_window_start).toLocaleString()} – ${new Date(gw.best_window_end!).toLocaleString()}`
                  : greenWindow.bestTime;
                const gridNow = gw?.current_intensity ?? greenWindow.normalIntensity;
                const gridBest = gw?.best_intensity ?? greenWindow.gridIntensity;
                const addSavings = gw?.savings_pct != null
                  ? `~${fmt(gw.savings_pct, 0)}% by scheduling`
                  : greenWindow.additionalSavings;
                return (
                  <>
                    <div className="p-4 rounded-lg bg-green-50 space-y-1">
                      <p className="text-sm font-medium text-foreground">Best time to run</p>
                      <p className="font-mono text-green-700 font-semibold">{bestTime}</p>
                    </div>
                    <div className="flex gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Grid now: </span>
                        <span className="font-mono font-semibold text-foreground">{fmt(gridNow, 0)} gCO₂/kWh</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">At best: </span>
                        <span className="font-mono font-semibold text-green-700">{fmt(gridBest, 0)} gCO₂/kWh</span>
                      </div>
                    </div>
                    <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0">
                      Additional savings: {addSavings}
                    </Badge>
                  </>
                );
              })()}
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
};

export default ResultsPage;
