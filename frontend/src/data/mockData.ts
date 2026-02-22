// Interfaces used by CodeDiff component and diff parser

export interface DiffLine {
  type: "removed" | "added" | "unchanged" | "hidden";
  content: string;
  oldLineNum?: number;
  newLineNum?: number;
  hiddenCount?: number;
}

export interface FileDiff {
  filename: string;
  optimization: string;
  co2Savings: string;
  lines: DiffLine[];
}

// Mock data for features not yet implemented in backend

export const mockMetrics = {
  water: { label: "Water Usage", unit: "liters", before: 3.2, after: 1.4, icon: "Droplets" },
  sci: { label: "SCI Score", unit: "points", before: 84, after: 35, icon: "Gauge" },
};

export const regionData = {
  current: { name: "US East (Virginia)", intensity: 383, unit: "gCO₂/kWh" },
  recommended: { name: "Canada (Montréal)", intensity: 24, unit: "gCO₂/kWh" },
  savingsPercent: 94,
};

export const greenWindow = {
  bestTime: "Saturday 2:00 AM – 5:00 AM EST",
  gridIntensity: 185,
  normalIntensity: 383,
  additionalSavings: "~1.2 kg CO₂e",
};
