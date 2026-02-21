// Central mock data file for GreenPull — easy to swap with real API later

export const pipelineSteps = [
  { id: 1, label: "Cloning repository", duration: 1800 },
  { id: 2, label: "Running baseline carbon measurement", duration: 2200 },
  { id: 3, label: "Scanning for optimization opportunities", duration: 2000 },
  { id: 4, label: "Generating optimization patches", duration: 2500 },
  { id: 5, label: "Re-running optimized version", duration: 2200 },
  { id: 6, label: "Calculating savings", duration: 1500 },
  { id: 7, label: "Preparing pull request", duration: 1800 },
];

export const metrics = {
  co2: { label: "CO₂e Emissions", unit: "kg", before: 12.4, after: 5.1, icon: "Cloud" },
  energy: { label: "Energy Usage", unit: "kWh", before: 8.7, after: 3.8, icon: "Zap" },
  water: { label: "Water Usage", unit: "liters", before: 3.2, after: 1.4, icon: "Droplets" },
  sci: { label: "SCI Score", unit: "points", before: 84, after: 35, icon: "Gauge" },
};

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

export const fileDiffs: FileDiff[] = [
  {
    filename: "train.py",
    optimization: "Mixed Precision (AMP)",
    co2Savings: "Saves ~4.2 kg CO₂e per run",
    lines: [
      { type: "unchanged", content: "import torch", oldLineNum: 1, newLineNum: 1 },
      { type: "unchanged", content: "import torch.nn as nn", oldLineNum: 2, newLineNum: 2 },
      { type: "removed", content: "from torch.optim import Adam", oldLineNum: 3 },
      { type: "added", content: "from torch.optim import Adam", newLineNum: 3 },
      { type: "added", content: "from torch.cuda.amp import autocast, GradScaler", newLineNum: 4 },
      { type: "unchanged", content: "", oldLineNum: 4, newLineNum: 5 },
      { type: "hidden", content: "", hiddenCount: 12 },
      { type: "unchanged", content: "def train_epoch(model, loader, optimizer):", oldLineNum: 17, newLineNum: 18 },
      { type: "removed", content: "    for batch in loader:", oldLineNum: 18 },
      { type: "removed", content: "        optimizer.zero_grad()", oldLineNum: 19 },
      { type: "removed", content: "        outputs = model(batch['input_ids'])", oldLineNum: 20 },
      { type: "removed", content: "        loss = outputs.loss", oldLineNum: 21 },
      { type: "removed", content: "        loss.backward()", oldLineNum: 22 },
      { type: "removed", content: "        optimizer.step()", oldLineNum: 23 },
      { type: "added", content: "    scaler = GradScaler()", newLineNum: 19 },
      { type: "added", content: "    for batch in loader:", newLineNum: 20 },
      { type: "added", content: "        optimizer.zero_grad()", newLineNum: 21 },
      { type: "added", content: "        with autocast():", newLineNum: 22 },
      { type: "added", content: "            outputs = model(batch['input_ids'])", newLineNum: 23 },
      { type: "added", content: "            loss = outputs.loss", newLineNum: 24 },
      { type: "added", content: "        scaler.scale(loss).backward()", newLineNum: 25 },
      { type: "added", content: "        scaler.step(optimizer)", newLineNum: 26 },
      { type: "added", content: "        scaler.update()", newLineNum: 27 },
      { type: "unchanged", content: "    return loss.item()", oldLineNum: 24, newLineNum: 28 },
    ],
  },
  {
    filename: "config.py",
    optimization: "INT8 Quantization",
    co2Savings: "Saves ~2.1 kg CO₂e per run",
    lines: [
      { type: "unchanged", content: "from transformers import AutoModelForCausalLM", oldLineNum: 1, newLineNum: 1 },
      { type: "removed", content: "from transformers import AutoTokenizer", oldLineNum: 2 },
      { type: "added", content: "from transformers import AutoTokenizer, BitsAndBytesConfig", newLineNum: 2 },
      { type: "unchanged", content: "", oldLineNum: 3, newLineNum: 3 },
      { type: "hidden", content: "", hiddenCount: 8 },
      { type: "unchanged", content: "def load_model(model_name):", oldLineNum: 12, newLineNum: 12 },
      { type: "removed", content: '    model = AutoModelForCausalLM.from_pretrained(model_name)', oldLineNum: 13 },
      { type: "added", content: "    quantization_config = BitsAndBytesConfig(", newLineNum: 13 },
      { type: "added", content: "        load_in_8bit=True,", newLineNum: 14 },
      { type: "added", content: "        llm_int8_threshold=6.0", newLineNum: 15 },
      { type: "added", content: "    )", newLineNum: 16 },
      { type: "added", content: "    model = AutoModelForCausalLM.from_pretrained(", newLineNum: 17 },
      { type: "added", content: "        model_name,", newLineNum: 18 },
      { type: "added", content: "        quantization_config=quantization_config,", newLineNum: 19 },
      { type: "added", content: '        device_map="auto"', newLineNum: 20 },
      { type: "added", content: "    )", newLineNum: 21 },
      { type: "unchanged", content: "    return model", oldLineNum: 14, newLineNum: 22 },
    ],
  },
  {
    filename: "train.py",
    optimization: "LoRA Fine-Tuning (PEFT)",
    co2Savings: "Saves ~1.8 kg CO₂e per run",
    lines: [
      { type: "unchanged", content: "from transformers import Trainer, TrainingArguments", oldLineNum: 25, newLineNum: 29 },
      { type: "added", content: "from peft import LoraConfig, get_peft_model, TaskType", newLineNum: 30 },
      { type: "unchanged", content: "", oldLineNum: 26, newLineNum: 31 },
      { type: "hidden", content: "", hiddenCount: 6 },
      { type: "unchanged", content: "def setup_training(model):", oldLineNum: 33, newLineNum: 38 },
      { type: "removed", content: "    # Full fine-tuning — all parameters trainable", oldLineNum: 34 },
      { type: "removed", content: "    for param in model.parameters():", oldLineNum: 35 },
      { type: "removed", content: "        param.requires_grad = True", oldLineNum: 36 },
      { type: "added", content: "    lora_config = LoraConfig(", newLineNum: 39 },
      { type: "added", content: "        task_type=TaskType.CAUSAL_LM,", newLineNum: 40 },
      { type: "added", content: "        r=16, lora_alpha=32,", newLineNum: 41 },
      { type: "added", content: '        target_modules=["q_proj", "v_proj"],', newLineNum: 42 },
      { type: "added", content: "        lora_dropout=0.05,", newLineNum: 43 },
      { type: "added", content: "    )", newLineNum: 44 },
      { type: "added", content: "    model = get_peft_model(model, lora_config)", newLineNum: 45 },
      { type: "unchanged", content: "    return model", oldLineNum: 37, newLineNum: 46 },
    ],
  },
];

export const energyBreakdown = [
  { name: "GPU", before: 5.8, after: 2.4, color: "hsl(142, 76%, 36%)" },
  { name: "CPU", before: 1.9, after: 0.9, color: "hsl(199, 89%, 48%)" },
  { name: "RAM", before: 1.0, after: 0.5, color: "hsl(262, 83%, 58%)" },
];

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
