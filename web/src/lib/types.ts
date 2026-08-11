export type ModelKey = "baseline" | "lstm" | "transformer";

export type SentimentLabel = "positive" | "negative";

export interface PredictionResult {
  label: SentimentLabel;
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
  };
  latency_ms: number;
}

export interface PredictResponse {
  success: boolean;
  model: string;
  prediction: PredictionResult;
}

export interface HealthResponse {
  status: string;
  models_available: string[];
  models_registered: string[];
}

export interface MetricsEntry {
  model: string;
  metrics: Record<string, number | string> | null;
}

export interface AllMetricsResponse {
  models: MetricsEntry[];
}

export interface ModelInfo {
  key: ModelKey;
  available: boolean;
}

export const MODEL_META: Record<
  ModelKey,
  { label: string; blurb: string }
> = {
  baseline: { label: "Baseline", blurb: "TF-IDF + LogReg" },
  lstm: { label: "LSTM", blurb: "Keras sequence model" },
  transformer: { label: "DistilBERT", blurb: "ONNX transformer" },
};

export const SAMPLE_REVIEWS = [
  {
    id: "amazing",
    label: "Positive",
    text: "This movie was absolutely amazing! The acting was superb, the storyline was gripping, and I was on the edge of my seat the entire time. Definitely one of the best films I've seen this year!",
  },
  {
    id: "terrible",
    label: "Negative",
    text: "What a waste of time. The plot was predictable, the acting was wooden, and I almost fell asleep halfway through. Save your money and skip this one.",
  },
  {
    id: "mixed",
    label: "Mixed",
    text: "It was okay, I guess. Some parts were interesting but overall it felt like a generic movie that didn't bring anything new to the table.",
  },
] as const;
