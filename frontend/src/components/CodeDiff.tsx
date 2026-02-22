import { useState, useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Leaf, ChevronDown, ChevronRight } from "lucide-react";
import type { FileDiff, DiffLine } from "@/data/mockData";

const CONTEXT_LINES = 3;

type Pair = { left: DiffLine | null; right: DiffLine | null };
type Section =
  | { kind: "lines"; pairs: Pair[] }
  | { kind: "collapsed"; pairs: Pair[]; count: number };

const CodeDiff = ({ diff }: { diff: FileDiff }) => {
  const [fileExpanded, setFileExpanded] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());

  const pairs = useMemo(() => {
    const result: Pair[] = [];
    let i = 0;
    const lines = diff.lines;
    while (i < lines.length) {
      const line = lines[i];
      if (line.type === "unchanged" || line.type === "hidden") {
        result.push({ left: line, right: line });
        i++;
      } else if (line.type === "removed") {
        const removed: DiffLine[] = [];
        while (i < lines.length && lines[i].type === "removed") removed.push(lines[i++]);
        const added: DiffLine[] = [];
        while (i < lines.length && lines[i].type === "added") added.push(lines[i++]);
        const max = Math.max(removed.length, added.length);
        for (let j = 0; j < max; j++) {
          result.push({ left: removed[j] || null, right: added[j] || null });
        }
      } else if (line.type === "added") {
        result.push({ left: null, right: line });
        i++;
      }
    }
    return result;
  }, [diff.lines]);

  // Split pairs into visible sections (context around changes) and collapsible sections
  const sections = useMemo(() => {
    const isChanged = pairs.map(
      (p) =>
        p.left?.type === "removed" ||
        p.right?.type === "added" ||
        p.left === null ||
        p.right === null ||
        p.left?.type === "hidden"
    );

    // Mark indices that should be visible (changed lines + N lines of context)
    const visible = new Array(pairs.length).fill(false);
    for (let i = 0; i < pairs.length; i++) {
      if (isChanged[i]) {
        for (let j = Math.max(0, i - CONTEXT_LINES); j <= Math.min(pairs.length - 1, i + CONTEXT_LINES); j++) {
          visible[j] = true;
        }
      }
    }

    // If everything is visible (small diff), show it all
    if (visible.every(Boolean)) {
      return [{ kind: "lines" as const, pairs }];
    }

    const result: Section[] = [];
    let i = 0;
    while (i < pairs.length) {
      if (visible[i]) {
        const chunk: Pair[] = [];
        while (i < pairs.length && visible[i]) chunk.push(pairs[i++]);
        result.push({ kind: "lines", pairs: chunk });
      } else {
        const chunk: Pair[] = [];
        while (i < pairs.length && !visible[i]) chunk.push(pairs[i++]);
        result.push({ kind: "collapsed", pairs: chunk, count: chunk.length });
      }
    }
    return result;
  }, [pairs]);

  // Stats for collapsed summary
  const addedCount = diff.lines.filter((l) => l.type === "added").length;
  const removedCount = diff.lines.filter((l) => l.type === "removed").length;

  const toggleSection = (idx: number) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const renderPair = (pair: Pair, key: string) => {
    if (pair.left?.type === "hidden") {
      return (
        <div key={key} className="text-center text-xs text-muted-foreground py-1.5 bg-muted/30 border-y border-dashed">
          ⋯ {pair.left.hiddenCount} lines hidden ⋯
        </div>
      );
    }
    return (
      <div key={key} className="flex">
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
        <div className="w-px bg-border flex-shrink-0" />
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
  };

  return (
    <div className="border rounded-lg overflow-hidden bg-card">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 bg-muted/50 border-b cursor-pointer"
        onClick={() => setFileExpanded(!fileExpanded)}
      >
        <div className="flex items-center gap-3">
          {fileExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <span className="font-mono text-sm font-medium text-foreground">{diff.filename}</span>
          <Badge variant="secondary" className="text-xs font-normal">{diff.optimization}</Badge>
          <span className="text-xs text-muted-foreground">
            <span className="text-green-600">+{addedCount}</span>{" "}
            <span className="text-red-500">−{removedCount}</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          {diff.co2Savings && (
            <div className="flex items-center gap-1.5 text-xs text-green-700 bg-green-50 px-2.5 py-1 rounded-full">
              <Leaf className="h-3 w-3" />
              {diff.co2Savings}
            </div>
          )}
          {!fileExpanded && (
            <span className="text-xs text-blue-600">Click to expand</span>
          )}
        </div>
      </div>

      {/* Diff body */}
      {fileExpanded && (
        <div className="overflow-x-auto text-[13px] leading-6 font-mono">
          {sections.map((section, sIdx) => {
            if (section.kind === "lines") {
              return (
                <div key={`sec-${sIdx}`}>
                  {section.pairs.map((pair, pIdx) => renderPair(pair, `${sIdx}-${pIdx}`))}
                </div>
              );
            }

            // Collapsed section
            if (expandedSections.has(sIdx)) {
              return (
                <div key={`sec-${sIdx}`}>
                  <button
                    onClick={() => toggleSection(sIdx)}
                    className="w-full text-center text-xs text-blue-600 hover:text-blue-800 py-1.5 bg-blue-50 hover:bg-blue-100 border-y border-dashed border-blue-200 cursor-pointer transition-colors"
                  >
                    ▴ Hide {section.count} unchanged lines
                  </button>
                  {section.pairs.map((pair, pIdx) => renderPair(pair, `${sIdx}-exp-${pIdx}`))}
                </div>
              );
            }

            return (
              <button
                key={`sec-${sIdx}`}
                onClick={() => toggleSection(sIdx)}
                className="w-full text-center text-xs text-blue-600 hover:text-blue-800 py-1.5 bg-blue-50/50 hover:bg-blue-100 border-y border-dashed border-blue-200/50 cursor-pointer transition-colors"
              >
                ⋯ Show {section.count} hidden lines ⋯
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CodeDiff;
