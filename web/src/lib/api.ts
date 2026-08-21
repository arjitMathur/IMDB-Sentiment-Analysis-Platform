import type {
  AllMetricsResponse,
  HealthResponse,
  ModelKey,
  PredictResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string; error?: string };
      detail = body.detail || body.error || detail;
    } catch {
      // keep default
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/v1/health");
}

export async function fetchMetrics(): Promise<AllMetricsResponse> {
  return apiFetch<AllMetricsResponse>("/api/v1/metrics");
}

export async function predict(
  text: string,
  model: ModelKey | "",
): Promise<PredictResponse> {
  return apiFetch<PredictResponse>("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify({ text, model }),
  });
}
