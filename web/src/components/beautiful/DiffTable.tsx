"use client";

import { MODEL_META, type ModelKey, type PredictionResult } from "@/lib/types";

/* Beautiful UI Diff Table — adapted for model comparison */

function formatPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function formatMs(n: number): string {
  return `${n.toFixed(1)} ms`;
}

export type DiffRow = {
  key: ModelKey;
  result: PredictionResult;
};

export default function DiffTable({
  title = "Model comparison",
  rows,
}: {
  title?: string;
  rows: DiffRow[];
}) {
  if (!rows.length) {
    return (
      <div className="w-full overflow-hidden rounded-card bg-surface shadow-card">
        <div className="primitive-card-bar flex items-center justify-between border-b border-line">
          <span className="text-[12.5px] font-medium text-ink">{title}</span>
        </div>
        <p className="primitive-card-pad text-[13px] text-ink-3">
          Run Compare all to see verdict, confidence, and latency across every
          trained model.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="relative overflow-hidden rounded-card bg-surface shadow-card">
        <div className="primitive-card-bar flex items-center justify-between border-b border-line">
          <span className="text-[12.5px] font-medium text-ink">{title}</span>
          <span className="text-[11px] text-ink-3">{rows.length} models</span>
        </div>

        <table className="w-full table-fixed border-collapse text-left">
          <colgroup>
            <col className="w-[28%]" />
            <col className="w-[22%]" />
            <col className="w-[25%]" />
            <col className="w-[25%]" />
          </colgroup>
          <thead>
            <tr className="border-b border-line">
              {["Model", "Verdict", "Confidence", "Latency"].map((h) => (
                <th
                  key={h}
                  className="primitive-table-cell text-[12px] font-medium text-ink-3"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ key, result }) => {
              const positive = result.label === "positive";
              return (
                <tr
                  key={key}
                  className="border-b border-line transition-colors duration-200 last:border-0 hover:bg-hover"
                  style={{
                    background: positive
                      ? "var(--green-tint)"
                      : "var(--red-tint)",
                  }}
                >
                  <td className="primitive-table-cell text-[13px] font-medium text-ink">
                    {MODEL_META[key].label}
                  </td>
                  <td
                    className="primitive-table-cell text-[13px] font-medium capitalize"
                    style={{
                      color: positive ? "var(--green)" : "var(--red)",
                    }}
                  >
                    {result.label}
                  </td>
                  <td className="primitive-table-cell font-mono text-[12.5px] text-ink-2 tabular-nums">
                    {formatPct(result.confidence)}
                  </td>
                  <td className="primitive-table-cell font-mono text-[12.5px] text-ink-2 tabular-nums">
                    {formatMs(result.latency_ms)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
