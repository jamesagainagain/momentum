const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export interface ActivatedTheme {
  theme_id: number;
  theme_name: string;
  score: number;
  weight: number;
  direction_score: number;
  direction_label: "up" | "down" | "flat";
  avg_volatility: number;
  avg_market_share: number;
  recent_momentum: number;
  threshold: number;
}

export interface TrendPoint {
  time_window: string;
  post_count: number;
  market_share: number;
  momentum: number;
  volatility: number;
}

export interface ClusterTrend {
  cluster_id: number;
  cluster_label: string;
  data: TrendPoint[];
}

export interface PredictionResponse {
  event_text: string;
  predicted_direction: "up" | "down" | "flat";
  predicted_direction_score: number;
  predicted_volatility: number;
  activated_theme_count: number;
  activated_themes: ActivatedTheme[];
  theme_scores: Record<string, number>;
  method: "heuristic" | "trained";
  confidence: number;
  class_probabilities: Record<string, number>;
  risk_band: "low" | "medium" | "high";
  risk_thresholds: { low: number; high: number };
  model_name?: string;
  heuristic_predicted_direction: string;
  heuristic_predicted_direction_score: number;
  activation_threshold: number;
  llm_semantic?: Record<string, unknown>;
  trend_data: ClusterTrend[];
  id?: string;
  timestamp?: string;
}

export interface ChatResponse {
  analysis: string;
  event_text: string;
}

export interface ClusterInfo {
  id: string;
  cluster: number;
  cluster_label: string;
  theme_name: string;
  size: number;
  market_share: number;
  volume_pct_change: number;
  volume_volatility: number;
  momentum: number;
  lifecycle: string;
  anomaly_score: number;
}

export interface Snapshot {
  cluster_id: string;
  cluster_label: string;
  theme_name?: string;
  time_window: string;
  post_count: number;
  market_share: number;
  momentum: number;
  volatility: number;
  window_type?: "1M" | "1W";
}

export interface TemporalEvent {
  date: string;
  cluster_label: string;
  event_type: string;
  description: string;
  sigma: number;
}

export interface KPIs {
  activeClusters: number;
  emergingCount: number;
  decliningCount: number;
  avgVolatility: number;
  eventSpikes: number;
}

export interface LifecycleItem {
  state: string;
  count: number;
}

export async function predictEvent(eventText: string): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ event_text: eventText }),
  });
}

export async function chatAnalysis(
  eventText: string,
  prediction: PredictionResponse,
  history: PredictionResponse[] = [],
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ event_text: eventText, prediction, history }),
  });
}

export async function fetchClusters(): Promise<{ clusters: ClusterInfo[]; latest_window: string; total: number }> {
  return request("/clusters");
}

export async function fetchSnapshots(): Promise<Snapshot[]> {
  return request("/snapshots");
}

export async function fetchTemporalEvents(): Promise<TemporalEvent[]> {
  return request("/temporal-events");
}

export async function fetchKPIs(): Promise<KPIs> {
  return request("/kpis");
}

export async function fetchLifecycleDistribution(): Promise<LifecycleItem[]> {
  return request("/lifecycle-distribution");
}

export async function fetchModelMetrics(): Promise<Record<string, unknown>> {
  return request("/model-metrics");
}

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean; themes: number; clusters: number }> {
  return request("/health");
}

export async function fetchWeeklySnapshots(): Promise<Snapshot[]> {
  return request("/snapshots/weekly-recent");
}
