"""
Export the trained linear models from car_price_model.pkl into a single
browser-friendly model.json so the Next.js web app can run LS / Ridge / Lasso
predictions entirely in JavaScript (no Python backend).

Key idea: LS and Ridge are raw dot products with their weight vectors. Lasso
runs on StandardScaler-transformed inputs, so we FOLD the scaler into effective
weights:
    pred = intercept + sum_j coef_j * (x_j - mean_j) / scale_j
         = [intercept - sum_j coef_j*mean_j/scale_j] + sum_j (coef_j/scale_j)*x_j
so the browser only needs a dot product on the RAW 606-feature vector.

Outputs:
    webapp/public/model.json   (created here; folder made if missing)

Also prints golden predictions for the default car for later JS parity checks.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "car_price_model.pkl")
CSV = os.path.join(HERE, "cleanedupcardataupdated_colors.csv")
OUT_DIR = os.path.join(HERE, "webapp", "public")
OUT = os.path.join(OUT_DIR, "model.json")

with open(PKL, "rb") as f:
    B = pickle.load(f)

features_full = list(B["feature_names_full"])
features_lasso = list(B["feature_names_lasso"])

ls_weights = np.asarray(B["ls_weights"], dtype=float).ravel()
ridge_weights = np.asarray(B["ridge_weights"], dtype=float).ravel()

# ----- Fold the StandardScaler into Lasso's effective weights -----
lasso = B["lasso_model"]
scaler = B["lasso_scaler"]
coef = np.asarray(lasso.coef_, dtype=float).ravel()           # (606,) in scaled units
intercept = float(np.asarray(lasso.intercept_).ravel()[0])
mean = np.asarray(scaler.mean_, dtype=float).ravel()          # (606,)
scale = np.asarray(scaler.scale_, dtype=float).ravel()        # (606,)

lasso_eff_weights = coef / scale
lasso_eff_intercept = intercept - float(np.sum(coef * mean / scale))

assert len(ls_weights) == len(features_full) == 607, len(ls_weights)
assert len(ridge_weights) == len(features_full)
assert len(lasso_eff_weights) == len(features_lasso) == 606, len(lasso_eff_weights)

# ----- Manufacturer -> models map, restricted to models the model knows -----
known_models = set(B["models"])
mm = pd.read_csv(CSV, usecols=["manufacturer_cat", "model_cat"]).drop_duplicates()
manufacturer_models = {}
for manu, grp in mm.groupby("manufacturer_cat"):
    models = sorted(m for m in grp["model_cat"].unique() if m in known_models)
    if models:
        manufacturer_models[str(manu)] = models

# Only keep manufacturers that actually have at least one known model
manufacturers = [m for m in B["manufacturers"] if m in manufacturer_models]

bundle = {
    "feature_names_full": features_full,
    "feature_names_lasso": features_lasso,
    "ls_weights": [round(float(x), 6) for x in ls_weights],
    "ridge_weights": [round(float(x), 6) for x in ridge_weights],
    "lasso_eff_weights": [round(float(x), 6) for x in lasso_eff_weights],
    "lasso_eff_intercept": round(lasso_eff_intercept, 6),
    "rmse": {
        "ls": float(B.get("rmse_ls", 8920.0)),
        "ridge": float(B.get("rmse_ridge", 8920.0)),
        "lasso": float(B.get("rmse_lasso", 8950.0)),
    },
    "r2": {
        "ls": 0.7850,
        "ridge": 0.7850,
        "lasso": 0.7836,
    },
    "manufacturers": manufacturers,
    "manufacturer_models": manufacturer_models,
    "fuel_types": list(B["fuel_types"]),
    "transmissions": list(B["transmissions"]),
    "drivetrains": list(B["drivetrains"]),
    "ext_colors": list(B["ext_colors"]),
    "int_colors": list(B["int_colors"]),
    "defaults": {
        "manufacturer": "BMW",
        "model": "X7",
        "fuel_type": B["default_fuel"],
        "transmission": B["default_transmission"],
        "drivetrain": "4WD",
        "ext_color": B["default_ext_color"],
        "int_color": B["default_int_color"],
        "year": 2025,
        "mileage": 30000,
        "accidents": False,
        "one_owner": True,
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(bundle, f, separators=(",", ":"))

size_kb = os.path.getsize(OUT) / 1024
print(f"Wrote {OUT}  ({size_kb:.1f} KB)")
print(f"manufacturers with models: {len(manufacturers)}")
print(f"total manufacturer->model pairs: {sum(len(v) for v in manufacturer_models.values())}")


# ----- Golden predictions for the default car (JS parity targets) -----
def predict(year, mileage, manufacturer, car_model, drivetrain, fuel_type,
            transmission, ext_color, int_color, accidents, one_owner):
    row = pd.DataFrame(0, index=[0], columns=features_full)
    row["intercept"] = 1
    row["year"] = year
    row["mileage"] = mileage
    row["accidents_or_damage"] = 1 if accidents else 0
    row["one_owner"] = 1 if one_owner else 0
    for col in [
        f"manufacturer_cat_{manufacturer}",
        f"model_cat_filt_{car_model}",
        f"drivetrain_cat_{drivetrain}",
        f"fuel_type_cat_{fuel_type}",
        f"transmission_cat_{transmission}",
        f"exterior_color_{ext_color}",
        f"interior_color_{int_color}",
    ]:
        if col in row.columns:
            row[col] = 1
    x_full = row[features_full].values.astype(float).ravel()
    x_lasso = row[features_lasso].values.astype(float).ravel()
    mu_ls = max(float(x_full @ ls_weights), 0)
    mu_ridge = max(float(x_full @ ridge_weights), 0)
    mu_lasso = max(lasso_eff_intercept + float(x_lasso @ lasso_eff_weights), 0)
    return mu_ls, mu_ridge, mu_lasso


d = bundle["defaults"]
cars = [
    (d["year"], d["mileage"], d["manufacturer"], d["model"], d["drivetrain"],
     d["fuel_type"], d["transmission"], d["ext_color"], d["int_color"],
     d["accidents"], d["one_owner"]),
    (2020, 30000, "BMW", "330", "4WD", "gasoline", "automatic", "Black", "Black", False, True),
    (2012, 120000, "Toyota", "Camry", "2WD", "gasoline", "automatic", "White", "Beige", True, False),
]
print("\nGOLDEN PREDICTIONS (LS, Ridge, Lasso):")
for c in cars:
    ls, rg, la = predict(*c)
    print(f"  {c[0]} {c[2]} {c[3]} {c[1]}mi acc={c[9]} owner={c[10]}: "
          f"LS=${ls:,.2f}  Ridge=${rg:,.2f}  Lasso=${la:,.2f}")
