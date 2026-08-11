"use client";

import { useState, type ReactNode } from "react";
import type { PredictionResult, SentimentLabel } from "@/lib/types";

/* Beautiful UI Recommendation Card — adapted for sentiment verdicts */

function Meter({ signal, tone }: { signal: number; tone: string }) {
  return (
    <span className="flex items-end gap-0.5">
      {[0, 1, 2].map((bar) => (
        <span
          key={bar}
          className="w-1 rounded-full transition-colors duration-300"
          style={{
            height: 10,
            background: bar < signal ? tone : "var(--line-strong)",
          }}
        />
      ))}
    </span>
  );
}

function formatPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function confidenceSignal(confidence: number): number {
  if (confidence >= 0.85) return 3;
  if (confidence >= 0.65) return 2;
  if (confidence >= 0.5) return 1;
  return 0;
}

function confidenceLabel(confidence: number): string {
  if (confidence >= 0.85) return "High confidence";
  if (confidence >= 0.65) return "Moderate confidence";
  if (confidence >= 0.5) return "Low confidence";
  return "No signal";
}

function toneFor(label: SentimentLabel): string {
  return label === "positive" ? "var(--green)" : "var(--red)";
}

export default function RecommendationCard({
  modelName,
  result,
  onCompare,
  comparing = false,
}: {
  modelName: string;
  result: PredictionResult;
  onCompare?: () => void;
  comparing?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const signal = confidenceSignal(result.confidence);
  const tone = toneFor(result.label);
  const alt: { key: string; body: ReactNode; short: string; label: string }[] =
    [
      {
        key: "pos",
        short: `Positive ${formatPct(result.probabilities.positive)}`,
        label:
          result.label === "positive" ? "Selected verdict" : "Alternate class",
        body: (
          <>
            Model leans{" "}
            <span className="font-medium text-green">positive</span> at{" "}
            <code className="rounded-md bg-green-tint px-1.5 py-0.5 font-mono text-[12px] text-green">
              {formatPct(result.probabilities.positive)}
            </code>
            .
          </>
        ),
      },
      {
        key: "neg",
        short: `Negative ${formatPct(result.probabilities.negative)}`,
        label:
          result.label === "negative" ? "Selected verdict" : "Alternate class",
        body: (
          <>
            Model leans{" "}
            <span className="font-medium text-red">negative</span> at{" "}
            <code className="rounded-md bg-red-tint px-1.5 py-0.5 font-mono text-[12px] text-red">
              {formatPct(result.probabilities.negative)}
            </code>
            .
          </>
        ),
      },
      {
        key: "latency",
        short: `${result.latency_ms.toFixed(1)} ms latency`,
        label: "Runtime",
        body: (
          <>
            Inference finished in{" "}
            <code className="rounded-md bg-accent-tint px-1.5 py-0.5 font-mono text-[12px] text-accent-ink">
              {result.latency_ms.toFixed(1)}_ms
            </code>{" "}
            on <span className="font-medium text-ink">{modelName}</span>.
          </>
        ),
      },
    ];

  return (
    <div
      className="w-full overflow-hidden rounded-card bg-surface shadow-card"
      aria-live="polite"
      style={{ animation: "fade-in 180ms ease-out both" }}
    >
      <div className="primitive-card-pad">
        <span className="text-[13px] font-semibold text-ink">
          Want me to call this review{" "}
          <span
            className="capitalize"
            style={{ color: tone }}
          >
            {result.label}
          </span>
          ?
        </span>
        <p className="mt-1.5 min-h-12 text-[13px] leading-relaxed text-ink-2">
          {modelName} scores{" "}
          <code className="rounded-md bg-accent-tint px-1.5 py-0.5 font-mono text-[12px] text-accent-ink">
            {formatPct(result.confidence)}
          </code>{" "}
          confidence with{" "}
          <code className="rounded-md bg-green-tint px-1.5 py-0.5 font-mono text-[12px] text-green">
            {formatPct(result.probabilities.positive)}
          </code>{" "}
          positive /{" "}
          <code className="rounded-md bg-red-tint px-1.5 py-0.5 font-mono text-[12px] text-red">
            {formatPct(result.probabilities.negative)}
          </code>{" "}
          negative.
        </p>
      </div>

      <div
        className="grid transition-[grid-template-rows,opacity] duration-300"
        style={{
          gridTemplateRows: open ? "1fr" : "0fr",
          opacity: open ? 1 : 0,
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
        }}
      >
        <div className="overflow-hidden">
          <div className="border-t border-line bg-inset px-2 py-2">
            <p className="px-1.5 pb-1 text-[11px] font-medium text-ink-3">
              Breakdown
            </p>
            {alt.map((o) => (
              <div
                key={o.key}
                className="flex w-full items-center gap-2.5 rounded-control px-1.5 py-1.5 text-left"
              >
                <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">
                  {o.short}
                </span>
                <span className="shrink-0 text-[11px] text-ink-3">{o.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="primitive-card-footer flex items-center justify-between gap-3 border-t border-line bg-inset">
        <span className="flex items-center gap-2">
          <Meter signal={signal} tone={tone} />
          <span className="text-[12.5px] font-medium text-ink-2">
            {confidenceLabel(result.confidence)}
          </span>
        </span>

        <span className="-mr-0.5 flex items-center gap-2">
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((current) => !current)}
            className={`h-7 rounded-control px-2.5 text-[12.5px] font-medium shadow-btn transition-[background-color,transform] duration-100 active:scale-[0.96] ${
              open ? "bg-hover text-ink" : "bg-surface text-ink hover:bg-hover"
            }`}
          >
            Breakdown
          </button>
          {onCompare && (
            <button
              type="button"
              onClick={onCompare}
              disabled={comparing}
              className="h-7 rounded-control bg-accent px-3 text-[12.5px] font-medium text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.14),0_0_0_1px_rgba(16,24,40,0.12),0_1px_2px_rgba(16,24,40,0.1)] transition-[background-color,transform] duration-150 active:scale-[0.96] disabled:opacity-50"
            >
              {comparing ? "Comparing…" : "Compare all"}
            </button>
          )}
        </span>
      </div>
    </div>
  );
}
