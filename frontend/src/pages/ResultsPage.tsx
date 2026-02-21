import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { metrics, fileDiffs, energyBreakdown, regionData, greenWindow } from "@/data/mockData";
import CodeDiff from "@/components/CodeDiff";
import { Leaf, GitPullRequest, Cloud, Zap, Droplets, Gauge, MapPin, Clock, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

const iconMap: Record<string, React.ElementType> = { Cloud, Zap, Droplets, Gauge };

const barData = Object.entries(metrics).map(([key, m]) => ({
  name: m.label.split(" ")[0],
  Before: m.before,
  After: m.after,
}));

const pieData = energyBreakdown.map((e) => ({ name: e.name, value: e.after }));

const chartConfig = {
  Before: { label: "Before", color: "hsl(0, 72%, 65%)" },
  After: { label: "After", color: "hsl(142, 76%, 36%)" },
};

const ResultsPage = () => {
  const navigate = useNavigate();
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
        {/* Metrics row */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(metrics).map(([key, m]) => {
            const Icon = iconMap[m.icon];
            const reduction = Math.round(((m.before - m.after) / m.before) * 100);
            return (
              <Card key={key} className="border">
                <CardContent className="p-5 space-y-2">
                  <div className="flex items-center gap-2 text-muted-foreground text-sm">
                    <Icon className="h-4 w-4" />
                    {m.label}
                  </div>
                  <div className="flex items-baseline gap-3">
                    <span className="text-2xl font-bold text-foreground">{m.after} <span className="text-sm font-normal text-muted-foreground">{m.unit}</span></span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground line-through">{m.before} {m.unit}</span>
                    <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-0 text-xs">
                      ↓ {reduction}%
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </section>

        {/* Code diffs */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold text-foreground">Optimization Diffs</h2>
          {fileDiffs.map((diff, i) => (
            <CodeDiff key={i} diff={diff} />
          ))}
        </section>

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

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Energy Breakdown (After)</CardTitle>
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
                      label={({ name, value }) => `${name}: ${value} kWh`}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={index} fill={energyBreakdown[index].color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Region & scheduling */}
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
