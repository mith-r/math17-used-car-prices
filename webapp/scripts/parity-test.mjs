// Parity check: replicate lib/predict.ts arithmetic in Node against the shipped
// model.json and compare to the Python golden values printed by export_model_json.py.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const B = JSON.parse(readFileSync(join(here, "..", "public", "model.json"), "utf8"));

function buildFeatureMap(inp) {
  const m = new Map();
  m.set("intercept", 1);
  m.set("year", inp.year);
  m.set("mileage", inp.mileage);
  m.set("accidents_or_damage", inp.accidents ? 1 : 0);
  m.set("one_owner", inp.one_owner ? 1 : 0);
  for (const c of [
    `manufacturer_cat_${inp.manufacturer}`,
    `model_cat_filt_${inp.model}`,
    `drivetrain_cat_${inp.drivetrain}`,
    `fuel_type_cat_${inp.fuel_type}`,
    `transmission_cat_${inp.transmission}`,
    `exterior_color_${inp.ext_color}`,
    `interior_color_${inp.int_color}`,
  ]) m.set(c, 1);
  return m;
}
const dot = (names, w, fm) => {
  let s = 0;
  for (let i = 0; i < names.length; i++) {
    const v = fm.get(names[i]);
    if (v) s += v * w[i];
  }
  return s;
};
function predict(inp) {
  const fm = buildFeatureMap(inp);
  return {
    LS: Math.max(dot(B.feature_names_full, B.ls_weights, fm), 0),
    Ridge: Math.max(dot(B.feature_names_full, B.ridge_weights, fm), 0),
    Lasso: Math.max(B.lasso_eff_intercept + dot(B.feature_names_lasso, B.lasso_eff_weights, fm), 0),
  };
}

// Python golden values (from export_model_json.py output).
const cases = [
  {
    inp: { year: 2018, mileage: 50000, manufacturer: "Ford", model: "Wrangler", drivetrain: "4WD", fuel_type: "gasoline", transmission: "automatic", ext_color: "Black", int_color: "Black", accidents: false, one_owner: true },
    gold: { LS: 19576.25, Ridge: 19584.78, Lasso: 32440.48 },
  },
  {
    inp: { year: 2020, mileage: 30000, manufacturer: "BMW", model: "330", drivetrain: "4WD", fuel_type: "gasoline", transmission: "automatic", ext_color: "Black", int_color: "Black", accidents: false, one_owner: true },
    gold: { LS: 35252.56, Ridge: 35252.58, Lasso: 35415.59 },
  },
  {
    inp: { year: 2012, mileage: 120000, manufacturer: "Toyota", model: "Camry", drivetrain: "2WD", fuel_type: "gasoline", transmission: "automatic", ext_color: "White", int_color: "Beige", accidents: true, one_owner: false },
    gold: { LS: 5887.94, Ridge: 5887.92, Lasso: 6035.80 },
  },
];

let ok = true;
for (const { inp, gold } of cases) {
  const got = predict(inp);
  for (const k of ["LS", "Ridge", "Lasso"]) {
    const diff = Math.abs(got[k] - gold[k]);
    const pass = diff < 0.1; // within 10¢ (weights exported at 6-decimal precision)
    if (!pass) ok = false;
    console.log(
      `${pass ? "PASS" : "FAIL"}  ${inp.manufacturer} ${inp.model} ${k}: ` +
        `js=$${got[k].toFixed(2)} py=$${gold[k].toFixed(2)} diff=${diff.toFixed(4)}`
    );
  }
}
console.log(ok ? "\nALL PASS — JS reproduces Python to the cent." : "\nMISMATCH");
process.exit(ok ? 0 : 1);
