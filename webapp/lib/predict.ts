// In-browser reproduction of app.py's LS / Ridge / Lasso predictions.
// All three linear models reduce to a dot product over the same one-hot feature
// vector. Lasso's StandardScaler was folded into effective weights at export time,
// so here every model is just: sum(weight_j * x_j) on the RAW feature vector.

export type ModelBundle = {
  feature_names_full: string[]; // 607, includes "intercept" at index 0
  feature_names_lasso: string[]; // 606, no intercept
  ls_weights: number[];
  ridge_weights: number[];
  lasso_eff_weights: number[];
  lasso_eff_intercept: number;
  rmse: { ls: number; ridge: number; lasso: number };
  r2: { ls: number; ridge: number; lasso: number };
  manufacturers: string[];
  manufacturer_models: Record<string, string[]>;
  fuel_types: string[];
  transmissions: string[];
  drivetrains: string[];
  ext_colors: string[];
  int_colors: string[];
  defaults: {
    manufacturer: string;
    model: string;
    fuel_type: string;
    transmission: string;
    drivetrain: string;
    ext_color: string;
    int_color: string;
    year: number;
    mileage: number;
    accidents: boolean;
    one_owner: boolean;
  };
};

export type CarInputs = {
  year: number;
  mileage: number;
  manufacturer: string;
  model: string;
  drivetrain: string;
  fuel_type: string;
  transmission: string;
  ext_color: string;
  int_color: string;
  accidents: boolean;
  one_owner: boolean;
};

export type ModelResult = {
  name: string;
  estimate: number; // clipped at 0
  rmse: number;
  r2: number;
  low: number; // 95% CI lower (clipped at 0)
  high: number; // 95% CI upper
};

// Build a featureName -> value map exactly like app.py:48-64.
function buildFeatureMap(inp: CarInputs): Map<string, number> {
  const m = new Map<string, number>();
  m.set("intercept", 1);
  m.set("year", inp.year);
  m.set("mileage", inp.mileage);
  m.set("accidents_or_damage", inp.accidents ? 1 : 0);
  m.set("one_owner", inp.one_owner ? 1 : 0);
  // One-hot columns: set to 1 if the column exists in the feature space.
  const onehot = [
    `manufacturer_cat_${inp.manufacturer}`,
    `model_cat_filt_${inp.model}`,
    `drivetrain_cat_${inp.drivetrain}`,
    `fuel_type_cat_${inp.fuel_type}`,
    `transmission_cat_${inp.transmission}`,
    `exterior_color_${inp.ext_color}`,
    `interior_color_${inp.int_color}`,
  ];
  for (const c of onehot) m.set(c, 1);
  return m;
}

function dot(names: string[], weights: number[], fm: Map<string, number>): number {
  let s = 0;
  for (let i = 0; i < names.length; i++) {
    const v = fm.get(names[i]);
    if (v) s += v * weights[i]; // missing or 0 contributes nothing
  }
  return s;
}

const Z = 1.96; // 95% CI

// --- Bringing the 2023-trained models up to the present ---
// The dataset is a single 2023 snapshot, so the models output 2023 prices for a
// given (model year, mileage). Two corrections express that in today's dollars:
//   1) Age. A model-year-Y car is DATA_LAG_YEARS further down its depreciation
//      curve now than in 2023. Feeding (year − DATA_LAG_YEARS) re-anchors it to
//      the model's 2023 frame, so each model applies its OWN year coefficient
//      (~$1,455/yr) as the aging rate.
//   2) Market level. Scale the dollars by the used-car-specific price index
//      (BLS CPI "Used cars and trucks", retail) since the data was collected.
//      Used-car prices fell after the 2022 peak, so this is BELOW 1 — unlike
//      general CPI, used cars are cheaper now than in 2023.
export const DATA_LAG_YEARS = 3; // 2026 − 2023 (data collected ~2023)
// BLS CPI "Used cars and trucks", retail, seasonally adjusted (FRED CUSR0000SETA02).
// Factor = latest index / dataset-period index, computed here so it's exact and auditable.
const CPI_USED_2023_AVG = 191.176; // 2023 monthly average (Jan–Dec 2023)
const CPI_USED_LATEST = 179.822; // April 2026 (latest published)
export const MARKET_FACTOR = CPI_USED_LATEST / CPI_USED_2023_AVG; // ≈ 0.9406

export function predict(B: ModelBundle, inp: CarInputs): ModelResult[] {
  // 1) Age the car relative to the 2023 snapshot.
  const aged: CarInputs = { ...inp, year: inp.year - DATA_LAG_YEARS };
  const fm = buildFeatureMap(aged);

  // 2) Re-level dollars to today's used-car market (scales estimate and band).
  const k = MARKET_FACTOR;
  const muLS = Math.max(dot(B.feature_names_full, B.ls_weights, fm) * k, 0);
  const muRidge = Math.max(dot(B.feature_names_full, B.ridge_weights, fm) * k, 0);
  const muLasso = Math.max(
    (B.lasso_eff_intercept + dot(B.feature_names_lasso, B.lasso_eff_weights, fm)) * k,
    0
  );

  const mk = (name: string, mu: number, rmse: number, r2: number): ModelResult => {
    const s = rmse * k; // band scales with the price level
    return {
      name,
      estimate: mu,
      rmse: s,
      r2,
      low: Math.max(mu - Z * s, 0),
      high: mu + Z * s,
    };
  };

  return [
    mk("Plain LS", muLS, B.rmse.ls, B.r2.ls),
    mk("Ridge", muRidge, B.rmse.ridge, B.r2.ridge),
    mk("Lasso", muLasso, B.rmse.lasso, B.r2.lasso),
  ];
}
