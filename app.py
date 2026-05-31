"""
Used Car Fair-Price Estimator — Math 17 Final Project demo

Shows each model's predicted PRICE DISTRIBUTION (not just point estimate),
plus 95% confidence intervals and Shannon entropy.

Usage:
    pip install gradio xgboost scikit-learn pandas numpy matplotlib scipy
    python app.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import gradio as gr

# ---------- Load bundle ----------
with open("car_price_model.pkl", "rb") as f:
    B = pickle.load(f)

FEATURES_FULL  = B["feature_names_full"]
FEATURES_LASSO = B["feature_names_lasso"]

# ---------- Noise model: residual std per model ----------
# Computed from the test-set RMSE of each model (which IS the empirical residual std)
# RMSE values come from the notebook — same as what's printed in the comparison table
RESIDUAL_STD = {
    "Plain LS":  B.get("rmse_ls",    8920.0),    # falls back if not saved
    "Ridge":     B.get("rmse_ridge", 8920.0),
    "Lasso":     B.get("rmse_lasso", 8950.0),
    "XGBoost":   B.get("rmse_xgb",   6910.0),
}


def gaussian_entropy_bits(sigma):
    """Differential entropy of Normal(mu, sigma^2), in bits."""
    return 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2)


def predict_with_distribution(year, mileage, manufacturer, car_model,
                              drivetrain, fuel_type, transmission,
                              ext_color, int_color, accidents, one_owner):
    """Return (markdown summary, plot figure) with distributions + entropy."""

    # Build feature row (with intercept) in FEATURES_FULL order
    row = pd.DataFrame(0, index=[0], columns=FEATURES_FULL)
    row["intercept"]           = 1
    row["year"]                = year
    row["mileage"]             = mileage
    row["accidents_or_damage"] = 1 if accidents else 0
    row["one_owner"]           = 1 if one_owner else 0
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

    X_full   = row[FEATURES_FULL].values.astype(float)
    X_no_int = row[FEATURES_LASSO].values.astype(float)

    # Point estimates
    mu = {
        "Plain LS":  float(X_full @ B["ls_weights"]),
        "Ridge":     float(X_full @ B["ridge_weights"]),
        "Lasso":     float(B["lasso_model"].predict(B["lasso_scaler"].transform(X_no_int))[0]),
        "XGBoost":   float(B["xgb_model"].predict(X_no_int.astype(np.float32))[0]),
    }
    # Clip negative predictions
    mu = {k: max(v, 0) for k, v in mu.items()}

    # Each model's Gaussian distribution: N(mu, sigma^2) with sigma = its test RMSE
    sigmas = RESIDUAL_STD

    # 95% CI and entropy
    ci = {m: (mu[m] - 1.96 * sigmas[m], mu[m] + 1.96 * sigmas[m]) for m in mu}
    entropy_bits = {m: gaussian_entropy_bits(sigmas[m]) for m in mu}

    # ----- Plot the four predictive distributions -----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 2])

    # Top panel: density curves
    colors = {"Plain LS": "#7f8c8d", "Ridge": "#3498db", "Lasso": "#e74c3c", "XGBoost": "#9b59b6"}
    # X-range spanning all distributions
    xmin = min(mu[m] - 4 * sigmas[m] for m in mu)
    xmax = max(mu[m] + 4 * sigmas[m] for m in mu)
    xx = np.linspace(max(xmin, 0), xmax, 400)
    for m in mu:
        pdf = stats.norm.pdf(xx, loc=mu[m], scale=sigmas[m])
        axes[0].plot(xx, pdf, label=f"{m} — μ=${mu[m]:,.0f}, σ=${sigmas[m]:,.0f}",
                     color=colors[m], linewidth=2)
        # Mark the mean with a vertical bar
        axes[0].axvline(mu[m], color=colors[m], linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Predicted price ($)")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title(f"Predictive distributions for: {year} {manufacturer} {car_model}, "
                      f"{mileage:,} mi")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # Bottom panel: 95% CIs as horizontal bars
    yticks = list(range(len(mu)))
    for i, m in enumerate(mu):
        low, high = ci[m]
        axes[1].plot([max(low, 0), high], [i, i], color=colors[m], linewidth=4, solid_capstyle="round")
        axes[1].plot(mu[m], i, "o", color=colors[m], markersize=10)
        axes[1].text(high + (xmax - xmin) * 0.01, i,
                     f"${mu[m]:,.0f}   (95% CI: ${max(low,0):,.0f}–${high:,.0f})",
                     va="center", fontsize=9)
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(list(mu.keys()))
    axes[1].set_xlabel("Price ($)")
    axes[1].set_title("95% confidence intervals — width = model uncertainty")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlim(max(xmin, 0), xmax * 1.15)

    plt.tight_layout()

    # ----- Markdown summary -----
    md = (
        f"### Car: {year} {manufacturer} {car_model} · {mileage:,} mi\n"
        f"{drivetrain} · {fuel_type} · {transmission}\n\n"
        f"#### Headline estimate (XGBoost, best model):\n"
        f"# **${mu['XGBoost']:,.0f}**\n"
        f"95% CI: ${max(ci['XGBoost'][0], 0):,.0f} – ${ci['XGBoost'][1]:,.0f}\n\n"
        f"#### All four models — point estimate, 95% CI, and Shannon entropy:\n"
        f"| Model | Estimate | 95% CI | σ (uncertainty) | Entropy |\n"
        f"|---|---|---|---|---|\n"
    )
    for m in mu:
        lo, hi = ci[m]
        md += (
            f"| {m} | **${mu[m]:,.0f}** | ${max(lo, 0):,.0f} – ${hi:,.0f} "
            f"| ${sigmas[m]:,.0f} | {entropy_bits[m]:.2f} bits |\n"
        )

    md += (
        f"\n*Lower entropy = more confident prediction. "
        f"XGBoost has the lowest entropy ({entropy_bits['XGBoost']:.2f} bits) "
        f"because it makes the tightest predictions; "
        f"Plain LS has the highest ({entropy_bits['Plain LS']:.2f} bits) "
        f"because its residuals have the largest spread.*\n\n"
        f"*Entropy of a Gaussian N(μ, σ²) = ½ log₂(2πe σ²). "
        f"Depends only on σ, the standard deviation of model residuals.*"
    )

    return md, fig


# ---------- UI ----------
with gr.Blocks(title="Used Car Fair-Price Estimator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Used Car Fair-Price Estimator (probabilistic)\n"
        "Each model produces a **predictive distribution**, not just a point estimate. "
        "Wider distribution = more uncertain. Entropy measures that uncertainty in bits."
    )

    with gr.Row():
        with gr.Column():
            year    = gr.Slider(1990, 2024, value=2018, step=1, label="Year")
            mileage = gr.Slider(0, 250_000, value=50_000, step=1000, label="Mileage")
            manufacturer = gr.Dropdown(B["manufacturers"], value=B["default_manufacturer"], label="Manufacturer")
            car_model    = gr.Dropdown(B["models"],        value=B["default_model"],        label="Model")
        with gr.Column():
            drivetrain   = gr.Dropdown(B["drivetrains"],   value=B["default_drivetrain"],   label="Drivetrain")
            fuel_type    = gr.Dropdown(B["fuel_types"],    value=B["default_fuel"],         label="Fuel type")
            transmission = gr.Dropdown(B["transmissions"], value=B["default_transmission"], label="Transmission")
            ext_color    = gr.Dropdown(B["ext_colors"],    value=B["default_ext_color"],    label="Exterior color")
            int_color    = gr.Dropdown(B["int_colors"],    value=B["default_int_color"],    label="Interior color")
        with gr.Column():
            accidents = gr.Checkbox(value=False, label="History of accidents or damage")
            one_owner = gr.Checkbox(value=True,  label="One previous owner")
            btn = gr.Button("Predict distribution", variant="primary", size="lg")

    with gr.Row():
        out_md = gr.Markdown()
        out_plot = gr.Plot()

    btn.click(
        predict_with_distribution,
        inputs=[year, mileage, manufacturer, car_model,
                drivetrain, fuel_type, transmission,
                ext_color, int_color, accidents, one_owner],
        outputs=[out_md, out_plot],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
