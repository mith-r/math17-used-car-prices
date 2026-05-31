"use client";

import { useEffect, useMemo, useState } from "react";
import {
  predict,
  DATA_LAG_YEARS,
  MARKET_FACTOR,
  type CarInputs,
  type ModelBundle,
  type ModelResult,
} from "@/lib/predict";

const MODEL_COLORS: Record<string, string> = {
  "Plain LS": "var(--ls)",
  Ridge: "var(--ridge)",
  Lasso: "var(--lasso)",
};

function fmt(n: number): string {
  return "$" + Math.round(n).toLocaleString("en-US");
}

export default function Page() {
  const [B, setB] = useState<ModelBundle | null>(null);
  const [inp, setInp] = useState<CarInputs | null>(null);

  // Load the exported model bundle once.
  useEffect(() => {
    fetch("/model.json")
      .then((r) => r.json())
      .then((data: ModelBundle) => {
        setB(data);
        const d = data.defaults;
        // Make sure the default model actually belongs to the default manufacturer
        // (the dataset's stored default can be an incoherent pair); otherwise fall
        // back to that manufacturer's first model so the dropdown stays in sync.
        const avail = data.manufacturer_models[d.manufacturer] ?? [];
        const model = avail.includes(d.model) ? d.model : avail[0] ?? d.model;
        setInp({
          year: d.year,
          mileage: d.mileage,
          manufacturer: d.manufacturer,
          model,
          drivetrain: d.drivetrain,
          fuel_type: d.fuel_type,
          transmission: d.transmission,
          ext_color: d.ext_color,
          int_color: d.int_color,
          accidents: d.accidents,
          one_owner: d.one_owner,
        });
      });
  }, []);

  // Models available for the currently-selected manufacturer (dependent dropdown).
  const modelOptions = useMemo(() => {
    if (!B || !inp) return [];
    return B.manufacturer_models[inp.manufacturer] ?? [];
  }, [B, inp]);

  const results: ModelResult[] = useMemo(() => {
    if (!B || !inp) return [];
    return predict(B, inp);
  }, [B, inp]);

  if (!B || !inp) {
    return (
      <div className="wrap">
        <p style={{ color: "var(--muted)" }}>Loading model…</p>
      </div>
    );
  }

  function set<K extends keyof CarInputs>(key: K, value: CarInputs[K]) {
    setInp((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  // When the manufacturer changes, reset model to that brand's first option.
  function setManufacturer(manufacturer: string) {
    const first = B!.manufacturer_models[manufacturer]?.[0] ?? "";
    setInp((prev) => (prev ? { ...prev, manufacturer, model: first } : prev));
  }

  // Shared x-range across all three models for the comparison bars.
  const lo = Math.min(...results.map((r) => r.low));
  const hi = Math.max(...results.map((r) => r.high));
  const span = Math.max(hi - lo, 1);
  const pct = (v: number) => ((v - lo) / span) * 100;

  // Average estimate as the headline "fair price".
  const headline = results.reduce((a, r) => a + r.estimate, 0) / results.length;

  return (
    <div className="wrap">
      <header>
        <h1>Used Car Fair-Price Estimator</h1>
        <p>
          Math 17 final project. Three regression models — Least Squares, Ridge,
          and Lasso — estimate a fair price from the car&apos;s details. The range
          on each is a 95% band from the model&apos;s test error.
        </p>
      </header>

      <div className="card">
        <div className="headline">
          <div className="est">{fmt(headline)}</div>
          <div className="sub">
            estimated fair price · {inp.year} {inp.manufacturer} {inp.model}
          </div>
        </div>

        {results.map((r) => {
          const left = pct(r.low);
          const width = Math.max(pct(r.high) - left, 1.5);
          const point = pct(r.estimate);
          const color = MODEL_COLORS[r.name];
          return (
            <div className="model" key={r.name}>
              <div className="model-top">
                <span className="model-name" style={{ color }}>
                  {r.name}
                </span>
                <span className="model-val">{fmt(r.estimate)}</span>
              </div>
              <div className="bar">
                <div
                  className="bar-range"
                  style={{ left: `${left}%`, width: `${width}%`, background: color }}
                />
                <div className="bar-point" style={{ left: `${point}%` }} />
              </div>
              <div className="model-ci">
                95% range {fmt(r.low)} – {fmt(r.high)} · R² {r.r2.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>

      <div className="card">
        <div className="row2">
          <div className="field">
            <label>Year</label>
            <input
              type="number"
              min={1990}
              max={2026}
              value={inp.year}
              onChange={(e) => set("year", Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>Mileage</label>
            <input
              type="number"
              min={0}
              max={400000}
              step={1000}
              value={inp.mileage}
              onChange={(e) => set("mileage", Number(e.target.value))}
            />
          </div>
        </div>

        <div className="field">
          <label>Manufacturer</label>
          <select
            value={inp.manufacturer}
            onChange={(e) => setManufacturer(e.target.value)}
          >
            {B.manufacturers.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>Model ({modelOptions.length} available)</label>
          <select value={inp.model} onChange={(e) => set("model", e.target.value)}>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="row2">
          <div className="field">
            <label>Drivetrain</label>
            <select
              value={inp.drivetrain}
              onChange={(e) => set("drivetrain", e.target.value)}
            >
              {B.drivetrains.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Fuel type</label>
            <select
              value={inp.fuel_type}
              onChange={(e) => set("fuel_type", e.target.value)}
            >
              {B.fuel_types.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="row2">
          <div className="field">
            <label>Transmission</label>
            <select
              value={inp.transmission}
              onChange={(e) => set("transmission", e.target.value)}
            >
              {B.transmissions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Exterior color</label>
            <select
              value={inp.ext_color}
              onChange={(e) => set("ext_color", e.target.value)}
            >
              {B.ext_colors.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="field">
          <label>Interior color</label>
          <select
            value={inp.int_color}
            onChange={(e) => set("int_color", e.target.value)}
          >
            {B.int_colors.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="toggles">
          <label className="toggle">
            <span>Accident / damage</span>
            <input
              type="checkbox"
              checked={inp.accidents}
              onChange={(e) => set("accidents", e.target.checked)}
            />
          </label>
          <label className="toggle">
            <span>One owner</span>
            <input
              type="checkbox"
              checked={inp.one_owner}
              onChange={(e) => set("one_owner", e.target.checked)}
            />
          </label>
        </div>
      </div>

      <div className="legend">
        Estimates update live as you change inputs. The models were trained on a
        2023 snapshot of listings, so each car is aged {DATA_LAG_YEARS} years
        down its depreciation curve and the result is scaled by the used-car
        price index (BLS CPI used cars &amp; trucks), about{" "}
        {Math.abs(Math.round((MARKET_FACTOR - 1) * 100))}%{" "}
        {MARKET_FACTOR < 1 ? "below" : "above"} its 2023 level, to express it in{" "}
        {2023 + DATA_LAG_YEARS} dollars. Typical error is a few thousand dollars,
        so treat the number as a ballpark, not an appraisal.
      </div>

      <div className="footer">Math 17 Final Project · Used-car regression</div>
    </div>
  );
}
