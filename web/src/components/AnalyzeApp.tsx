"use client";

import { useEffect, useState, useTransition } from "react";
import DiffTable from "@/components/beautiful/DiffTable";
import LoadingState from "@/components/beautiful/LoadingState";
import PromptBar from "@/components/beautiful/PromptBar";
import RecommendationCard from "@/components/beautiful/RecommendationCard";
import { fetchHealth, fetchMetrics, getApiBase, predict } from "@/lib/api";
import {
  MODEL_META,
  type AllMetricsResponse,
  type HealthResponse,
  type ModelKey,
  type PredictResponse,
} from "@/lib/types";

const MODEL_KEYS: ModelKey[] = ["baseline", "lstm", "transformer"];

function formatPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

export default function AnalyzeApp() {
  const [text, setText] = useState("");
  const [model, setModel] = useState<ModelKey>("baseline");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<AllMetricsResponse | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [comparison, setComparison] = useState<
    { key: ModelKey; response: PredictResponse }[]
  >([]);
  const [error, setError] = useState<string | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [comparing, setComparing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, m] = await Promise.all([fetchHealth(), fetchMetrics()]);
        if (cancelled) return;
        setHealth(h);
        setMetrics(m);
        const available = h.models_available as ModelKey[];
        if (available.length && !available.includes(model)) {
          setModel(available[0]);
        }
      } catch {
        if (!cancelled) {
          setBootError(
            `Cannot reach API at ${getApiBase()}. Start FastAPI with uvicorn api.main:app --reload`,
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- boot once
  }, []);

  const available = new Set(health?.models_available ?? []);
  const modelReady = available.has(model);
  const modelOptions = MODEL_KEYS.map((key) => ({
    key,
    available: available.has(key),
  }));

  function runAnalyze() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Enter a review to analyze.");
      return;
    }
    setError(null);
    setComparison([]);
    startTransition(async () => {
      try {
        const res = await predict(trimmed, model);
        setResult(res);
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "Prediction failed.");
      }
    });
  }

  async function runCompare() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Enter a review to compare models.");
      return;
    }
    setError(null);
    setComparing(true);
    try {
      const keys = MODEL_KEYS.filter((k) => available.has(k));
      if (!keys.length) {
        setError("No trained models available to compare.");
        return;
      }
      const responses = await Promise.all(
        keys.map(async (key) => ({
          key,
          response: await predict(trimmed, key),
        })),
      );
      setComparison(responses);
      if (!result && responses[0]) {
        setResult(
          responses.find((r) => r.key === model)?.response ??
            responses[0].response,
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setComparing(false);
    }
  }

  const benchmarkRows =
    metrics?.models
      ?.map((entry) => {
        const m = entry.metrics;
        if (!m) return null;
        const acc = m.test_accuracy ?? m.val_accuracy;
        const f1 = m.test_f1 ?? m.val_f1;
        const train = m.training_time_seconds;
        return {
          model: entry.model,
          accuracy: typeof acc === "number" ? formatPct(acc) : "-",
          f1: typeof f1 === "number" ? f1.toFixed(3) : "-",
          train: typeof train === "number" ? `${train.toFixed(1)} s` : "-",
        };
      })
      .filter(Boolean) ?? [];

  return (
    <div className="mx-auto flex w-full max-w-[720px] flex-1 flex-col px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-8 flex items-baseline justify-between gap-4 border-b border-line pb-5">
        <div>
          <p className="text-lg font-medium tracking-tight text-ink">ReelTone</p>
          <p className="mt-1 text-sm text-ink-3">Movie review sentiment</p>
        </div>
        <a
          href={`${getApiBase()}/docs`}
          target="_blank"
          rel="noreferrer"
          className="text-sm text-ink-3 transition-colors hover:text-ink"
        >
          API docs
        </a>
      </header>

      <main className="flex flex-1 flex-col gap-6">
        <div>
          <h1 className="text-2xl font-medium tracking-tight text-ink sm:text-3xl">
            Analyze sentiment in any review
          </h1>
          <p className="mt-2 max-w-[55ch] text-sm leading-relaxed text-ink-2">
            Paste a review, pick a model, and compare how baseline, LSTM, and
            DistilBERT disagree.
          </p>
        </div>

        {bootError && (
          <p
            role="alert"
            className="rounded-card border border-line bg-surface px-4 py-3 text-sm text-ink shadow-card"
          >
            {bootError}
          </p>
        )}

        <PromptBar
          value={text}
          onChange={setText}
          model={model}
          onModelChange={setModel}
          models={modelOptions}
          onSubmit={runAnalyze}
          onCompare={runCompare}
          disabled={!!bootError || isPending}
        />

        <div className="flex flex-wrap items-center justify-between gap-3 text-[12px] text-ink-3">
          <span className="font-mono tabular-nums">
            {text.length} / 5000
          </span>
          {health && !modelReady && (
            <span>
              {MODEL_META[model].label} is not trained yet.{" "}
              <code className="font-mono text-ink">
                python -m src.sentiment.train {model}
              </code>
            </span>
          )}
        </div>

        {error && (
          <p role="alert" className="text-sm text-red">
            {error}
          </p>
        )}

        {(isPending || comparing) && (
          <LoadingState
            label={comparing ? "Comparing" : "Analyzing"}
            variant="Drive"
            running
          />
        )}

        {result && !isPending && (
          <RecommendationCard
            modelName={result.model}
            result={result.prediction}
            onCompare={runCompare}
            comparing={comparing}
          />
        )}

        <DiffTable
          rows={comparison.map(({ key, response }) => ({
            key,
            result: response.prediction,
          }))}
        />

        <section className="overflow-hidden rounded-card bg-surface shadow-card">
          <div className="primitive-card-bar flex items-center justify-between border-b border-line">
            <span className="text-[12.5px] font-medium text-ink">
              Benchmarks
            </span>
          </div>
          {!benchmarkRows.length ? (
            <p className="primitive-card-pad text-[13px] text-ink-3">
              No benchmark metrics yet. Train models to populate accuracy and
              F1.
            </p>
          ) : (
            <table className="w-full table-fixed border-collapse text-left">
              <thead>
                <tr className="border-b border-line">
                  {["Model", "Accuracy", "F1", "Train time"].map((h) => (
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
                {benchmarkRows.map((row) =>
                  row ? (
                    <tr
                      key={row.model}
                      className="border-b border-line last:border-0 hover:bg-hover"
                    >
                      <td className="primitive-table-cell capitalize text-[13px] text-ink">
                        {row.model}
                      </td>
                      <td className="primitive-table-cell font-mono text-[12.5px] text-ink-2">
                        {row.accuracy}
                      </td>
                      <td className="primitive-table-cell font-mono text-[12.5px] text-ink-2">
                        {row.f1}
                      </td>
                      <td className="primitive-table-cell font-mono text-[12.5px] text-ink-2">
                        {row.train}
                      </td>
                    </tr>
                  ) : null,
                )}
              </tbody>
            </table>
          )}
        </section>
      </main>

      <footer className="mt-auto border-t border-line pt-5 text-xs text-ink-3">
        ReelTone - local FastAPI at {getApiBase()}
      </footer>
    </div>
  );
}
