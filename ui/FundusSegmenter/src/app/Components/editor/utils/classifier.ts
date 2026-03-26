/**
 * Client-side logistic regression inference.
 *
 * Loads exported StandardScaler + LogisticRegression weights from JSON
 * and computes predictions in pure TypeScript — no runtime dependencies.
 *
 * Pipeline: z = (x - mean) / scale → p = σ(z · coef + intercept)
 */

import { FeatureEntry, METRICS, LESION_IDS, LESION_NAMES } from './features';
import { REGION_ORDER } from './regions';

// ── model definition ────────────────────────────────────────────────────

export interface LogisticModelWeights {
  scaler_mean: number[];
  scaler_scale: number[];
  coef: number[];
  intercept: number;
  threshold: number;
  feature_names: string[];
}

export interface Prediction {
  /** Probability of positive class (referable DR). */
  probability: number;
  /** Binary prediction at the configured threshold. */
  positive: boolean;
  /** The threshold used. */
  threshold: number;
}

// ── singleton model holder ──────────────────────────────────────────────

let _model: LogisticModelWeights | null = null;

let _score_model: LogisticModelWeights | null = null;

export async function loadScoreModel(
  url: string,
): Promise<LogisticModelWeights> {
  const resp = await fetch(url);
  if (!resp.ok)
    throw new Error(`Failed to load score model: ${resp.statusText}`);
  _score_model = (await resp.json()) as LogisticModelWeights;

  // Basic validation
  const n = _score_model.feature_names.length;
  if (_score_model.scaler_mean.length !== n)
    throw new Error(`score_model scaler_mean length mismatch`);
  if (_score_model.scaler_scale.length !== n)
    throw new Error(`score_model scaler_scale length mismatch`);
  if (_score_model.coef.length !== n) {
    const msg = `score_model coef length mismatch, got ${_score_model.coef.length}, expected ${n}`;
    throw new Error(msg);
  }

  return _score_model;
}
/**
 * Load model weights from a JSON asset.
 * Call once at app startup or lazily before first prediction.
 */
export async function loadModel(url: string): Promise<LogisticModelWeights> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load model: ${resp.statusText}`);
  _model = (await resp.json()) as LogisticModelWeights;

  // Basic validation
  const n = _model.feature_names.length;
  if (_model.scaler_mean.length !== n)
    throw new Error(`scaler_mean length mismatch`);
  if (_model.scaler_scale.length !== n)
    throw new Error(`scaler_scale length mismatch`);
  if (_model.coef.length !== n) throw new Error(`coef length mismatch`);

  return _model;
}

export function getModel(): LogisticModelWeights | null {
  return _model;
}
export function getScoreModel(): LogisticModelWeights | null {
  return _score_model;
}

// ── inference ───────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  // Numerically stable for large negative values
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Convert the 48-entry FeatureEntry[] into a flat number[] aligned
 * with the model's expected feature order.
 *
 * The canonical order is:
 *   for metric in [count, total_area, mean_area]:
 *     for region in [OD, macula, 1OD-2OD, elsewhere]:
 *       for lesion in [CWS, EX, HEM, MA]:
 *         value
 */
export function featureVectorToArray(features: FeatureEntry[]): number[] {
  const arr = new Array(48).fill(0);
  let i = 0;
  for (const metric of METRICS) {
    for (const region of REGION_ORDER) {
      for (const lid of LESION_IDS) {
        const entry = features.find(
          (f) =>
            f.metric === metric &&
            f.region === region &&
            f.lesion === LESION_NAMES[lid],
        );
        arr[i++] = entry?.value ?? 0;
      }
    }
  }
  return arr;
}

/**
 * Run logistic regression prediction on a feature vector.
 */
export function predict(
  features: FeatureEntry[],
  model?: LogisticModelWeights,
): Prediction {
  const m = model ?? _model;
  if (!m) throw new Error('Model not loaded. Call loadModel() first.');

  const x = featureVectorToArray(features);

  // StandardScaler: z = (x - mean) / scale
  // Logistic: logit = z · coef + intercept
  let logit = m.intercept;
  for (let i = 0; i < x.length; i++) {
    const z = (x[i] - m.scaler_mean[i]) / m.scaler_scale[i];
    logit += z * m.coef[i];
  }

  const probability = sigmoid(logit);
  return {
    probability,
    positive: probability >= m.threshold,
    threshold: m.threshold,
  };
}

export function predictScore(
  features: FeatureEntry[],
  model?: LogisticModelWeights,
): number {
  const m = model ?? _score_model;
  if (!m)
    throw new Error('Score model not loaded. Call loadScoreModel() first.');

  const x = featureVectorToArray(features);

  // StandardScaler: z = (x - mean) / scale
  // Logistic: logit = z · coef + intercept
  let logit = m.intercept;
  for (let i = 0; i < x.length; i++) {
    const z = (x[i] - m.scaler_mean[i]) / m.scaler_scale[i];
    logit += z * m.coef[i];
  }

  return logit;
}
