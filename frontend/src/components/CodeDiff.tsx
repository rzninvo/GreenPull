import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Leaf, ChevronDown, ChevronRight } from "lucide-react";
import type { FileDiff, DiffLine } from "@/data/mockData";

const CodeDiff = ({ diff }: { diff: FileDiff }) => {
  const [expanded, setExpanded] = useState(true);

  // Build left/right line pairs for side-by-side display
  const buildPairs = (lines: DiffLine[]) => {
    const pairs: { left: DiffLine | null; right: DiffLine | null }[] = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.type === "unchanged" || line.type === "hidden") {
        pairs.push({ left: line, right: line });
        i++;
      } else if (line.type === "removed") {
        // Collect consecutive removed, then consecutive added
        const removed: DiffLine[] = [];
        while (i < lines.length && lines[i].type === "removed") removed.push(lines[i++]);
        const added: DiffLine[] = [];
        while (i < lines.length && lines[i].type === "added") added.push(lines[i++]);
        const max = Math.max(removed.length, added.length);
        for (let j = 0; j < max; j++) {
          pairs.push({ left: removed[j] || null, right: added[j] || null });
        }
      } else if (line.type === "added") {
        pairs.push({ left: null, right: line });
        i++;
      }
    }
    return pairs;
  };

  const pairs = buildPairs(diff.lines);

  return (
    <div className="border rounded-lg overflow-hidden bg-card">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 bg-muted/50 border-b cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {expanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
          <span className="font-mono text-sm font-medium text-foreground">{diff.filename}</span>
          <Badge variant="secondary" className="text-xs font-normal">{diff.optimization}</Badge>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-green-700 bg-green-50 px-2.5 py-1 rounded-full">
          <Leaf className="h-3 w-3" />
          {diff.co2Savings}
        </div>
      </div>

      {/* Diff body */}
      {expanded && (
        <div className="overflow-x-auto text-[13px] leading-6 font-mono">
          {pairs.map((pair, idx) => {
            if (pair.left?.type === "hidden") {
              return (
                <div key={idx} className="text-center text-xs text-muted-foreground py-2 bg-muted/30 border-y border-dashed">
                  ⋯ {pair.left.hiddenCount} lines hidden ⋯
                </div>
              );
            }
            return (
              <div key={idx} className="flex">
                {/* Left (old) */}
                <div
                  className={`flex-1 flex min-w-0 ${
                    pair.left?.type === "removed"
                      ? "bg-[hsl(0,100%,95%)]"
                      : pair.left === null
                      ? "bg-muted/20"
                      : ""
                  }`}
                >
                  <span className="w-10 flex-shrink-0 text-right pr-3 text-muted-foreground/50 select-none">
                    {pair.left?.oldLineNum ?? ""}
                  </span>
                  <span className={`flex-1 pr-4 ${pair.left?.type === "removed" ? "text-red-800" : ""}`}>
                    {pair.left?.type === "removed" && <span className="text-red-400 mr-1">−</span>}
                    {pair.left?.content ?? ""}
                  </span>
                </div>
                {/* Divider */}
                <div className="w-px bg-border flex-shrink-0" />
                {/* Right (new) */}
                <div
                  className={`flex-1 flex min-w-0 ${
                    pair.right?.type === "added"
                      ? "bg-[hsl(120,60%,95%)]"
                      : pair.right === null
                      ? "bg-muted/20"
                      : ""
                  }`}
                >
                  <span className="w-10 flex-shrink-0 text-right pr-3 text-muted-foreground/50 select-none">
                    {pair.right?.newLineNum ?? ""}
                  </span>
                  <span className={`flex-1 pr-4 ${pair.right?.type === "added" ? "text-green-800" : ""}`}>
                    {pair.right?.type === "added" && <span className="text-green-500 mr-1">+</span>}
                    {pair.right?.content ?? ""}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CodeDiff;
