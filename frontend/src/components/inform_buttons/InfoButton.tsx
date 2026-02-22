import { useState, useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";

const InfoButton = () => {
  const [expanded, setExpanded] = useState(false);
  const sciHtml = useMemo(() => katex.renderToString("SCI = \\frac{(E \\times I) + M}{R}", { throwOnError: false, displayMode: true }), []);
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="fixed top-4 left-4 z-50 rounded-full border border-border bg-background shadow-sm hover:bg-accent"
          aria-label="Information"
        >
          <Info className="h-5 w-5 text-muted-foreground" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-foreground">Why Green Software?</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
          <p>
            The explosive growth of AI and cloud computing is consuming so much energy and water that it threatens the planet, forcing the tech industry to urgently switch to building energy-efficient "Green Software" to survive strict new environmental laws and keep investor funding.
          </p>

          <Collapsible open={expanded} onOpenChange={setExpanded}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="w-full gap-2 text-sm">
                {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                {expanded ? "Hide Details" : "Read More Explanation"}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-4 space-y-4 text-sm leading-relaxed text-muted-foreground">
              <p>
                The global IT sector is causing massive environmental damage through its raw resource consumption. U.S. data centers emitted 105 million metric tons of CO₂e in 2024, representing 2% of total national emissions. Globally, data center electricity consumption reached 460 TWh in 2022 and is projected to hit 1,050 TWh by 2026. Artificial intelligence drastically accelerates this drain. Training a single large model can emit 550 metric tons of CO₂e, while routine inference accounts for up to 90% of a model's total energy cost. A single generative AI query consumes 10 times more electricity than a standard search, and AI cooling systems are projected to drain up to 1,125 million cubic meters of water annually by 2030.
              </p>
              <p>
                The industry is fighting this by tracking exact emissions through the Software Carbon Intensity metric. Simple optimizations yield massive results, such as switching away from standard Python – which uses 50 times more energy than C – to cut energy overhead in half. New laws like the EU's CSRD now force over 50,000 companies to legally report their complete digital carbon footprint. Because unchecked software expansion is unsustainable, global capital is shifting rapidly. A recent survey showed only 20% of investors still back AI tech giants, while over 50% are now investing strictly in the energy providers powering the grid.
              </p>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InfoButton;
