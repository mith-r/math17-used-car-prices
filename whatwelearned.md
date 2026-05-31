# What We Learned

Running log of insights, surprises, and challenges from building the regression models. These are the things worth talking about in the final presentation.

---

## Plain LS — baseline with 4 numeric features only

**Setup**
- Features: `year`, `mileage`, `accidents_or_damage`, `one_owner` (no categoricals yet)
- Train/test split: 80/20 on 642,445 rows, `random_state=1`
- Solved via the normal equation: `w = np.linalg.solve(X.T @ X, X.T @ y)`

**Learned weights**

| Feature | Weight | Expected sign | Verdict |
|---|---|---|---|
| intercept | +138,199.24 | — | (just shifts the line) |
| year | **−53.62** | should be positive | **wrong sign** |
| mileage | **+0.02** | should be negative | **wrong sign** |
| accidents_or_damage | −14,482.14 | negative | ✓ |
| one_owner | +18,225.87 | positive | ✓ (but too large) |

### Challenge 1 — Year and mileage have the wrong sign

The model says newer cars are *cheaper* (year coefficient negative) and high-mileage cars are *more expensive* (mileage coefficient positive). Both contradict common sense.

This is a textbook diagnostic of two underlying problems:

1. **Multicollinearity.** Year and mileage are heavily negatively correlated (newer cars have less mileage). When two predictors carry overlapping information, plain OLS can split credit between them in nonsensical ways, including flipping signs. Our PCA from earlier confirmed this — year and mileage loaded together on PC1 with opposite signs.

2. **Missing features.** With no brand or trim information, the model can't distinguish a 2018 BMW from a 2018 Honda Civic. Year and mileage end up forced to carry signal that properly belongs to manufacturer, and they get distorted in the process.

### Challenge 2 — `one_owner` coefficient is unrealistically large

A binary "one owner" flag is worth +$18,226 in the model. That's far too large for a single yes/no feature.

What's happening: with no manufacturer feature in the model, `one_owner` is absorbing signal that should belong to brand and trim. Someone who buys a $60k Tesla holds it longer; someone who buys a $5k Civic flips it sooner. So `one_owner = 1` correlates strongly with "this is a nicer car," and the coefficient inflates to match. We expect this weight to drop to $1–3k once we add manufacturer.

### Baseline test metrics

| Metric | Value |
|---|---|
| Test MSE | 607,649,493 |
| Test RMSE | **$24,650.55** |
| Test R² | **−0.1009** |

### Challenge 3 — Negative R²: the baseline is worse than predicting the mean

R² = 1 − (SSR / SST). When R² is negative, the sum of squared residuals exceeds the total variance in the test prices. Translated: this trained model does **worse than the dumbest possible baseline of predicting the mean price for every car**.

That's a direct consequence of Challenges 1 and 2. The wrong signs on `year` and `mileage` aren't merely suboptimal — they're flipping predictions in the wrong direction. A 2020 Tesla gets predicted cheap; a beat-up 2005 sedan gets predicted expensive. Those huge inverted residuals pile up and overwhelm the variance of just guessing the average price.

The RMSE of ~$24,650 confirms it. Average used car prices in this dataset are roughly $20–30k, so the typical prediction error is on the order of the price itself. The model is essentially noise.

### The takeaway for the presentation

A plain LS model trained on too few features produces coefficients that **flatly contradict common sense**, and it predicts **worse than no model at all** — even though the math is correct. The model isn't broken — it's *underfit* and *confounded*. This sets up the project's central question and gives us a dramatic "before" picture: once we add categoricals (and apply Ridge/Lasso), R² should jump from negative to positive in a single step.

---

## Plain LS — outlier discovery

### Challenge 4 — Adding more features made the model WORSE

The natural next step was to add categorical features (manufacturer, fuel type, transmission, drivetrain) via one-hot encoding. Expected R² to jump from −0.10 to something like 0.6.

Instead, R² collapsed from −0.10 to **−0.85**. The model with more information was *more wrong* than the model with less. Training R² for the full model was essentially zero (0.0001).

### The diagnosis

The dataset had extreme price outliers we hadn't noticed:

- 3 listings priced at **$1,000,000,000** (a 2009 Chevy Cobalt and two 2018 Dodge Durangos — typos with extra zeros)
- 13 listings over $1M (mostly suspicious low-mileage Porsche Carreras)
- 32 over $500k, 1,028 over $200k
- 1 listing at $1 (an Acura TLX, almost certainly a placeholder)
- 26 total listings under $1,000 (likely junk or "call for price" entries)

For the baseline 4-feature model, none of the features could explain a $1B price, so it just predicted near the mean and got mediocre-but-not-catastrophic results. The 40-feature model had enough flexibility to chase those outliers via specific manufacturer dummies — and doing so distorted every prediction for those brands, blowing up test error.

### The fix

Filtered the price column to `$500 ≤ price ≤ $200,000`. This dropped 1,039 rows (0.16% of the data) — including all the $1B listings, the $1 placeholder, and the suspicious $1M+ entries. The 99.9th percentile was $230k, so the $200k cap stays close to the natural distribution.

### Baseline numbers after filtering (cleaned)

| Model | Features | Test R² | Test RMSE |
|---|---|---|---|
| Plain LS — 4 numeric only | 5 (incl. intercept) | **0.2956** | $16,148 |
| Plain LS — 4 numeric + 4 categoricals | 40 | **0.5441** | $12,991 |

R² jumped from 0.30 → 0.54 by adding categoricals. RMSE dropped by ~$3,200 per car.

### Challenge 5 — Coefficients went from broken to interpretable

After filtering outliers and adding categoricals, every sign and magnitude makes intuitive sense:

| Feature | 4-feature model | 40-feature model | Change |
|---|---|---|---|
| year | **−$54** (wrong sign) | **+$1,092** (correct) | flipped |
| mileage | **+$0.02** (wrong sign) | **−$0.14** (correct) | flipped |
| accidents_or_damage | −$14,482 | −$2,656 | realistic magnitude |
| one_owner | +$18,226 | +$695 | deflated 26× — no longer absorbing brand signal |

Manufacturer dummies (relative to the dropped reference, alphabetically *Acura*):

- **Most expensive premium**: Porsche +$40k, Land Rover +$17k, Mercedes-Benz +$13k, Cadillac +$9k, Tesla +$9k, Lexus +$7k, BMW +$6k
- **Largest discount**: Mitsubishi −$11k, Subaru −$9k, Hyundai/Mazda/Kia/VW all around −$7k

The ordering matches what any used-car shopper would expect. A perfect "the model now agrees with common sense" slide.

### The fuel-type interpretation trap

```
fuel_type_cat_e85       −$20,379
fuel_type_cat_gasoline  −$19,131
fuel_type_cat_hybrid    −$18,100
fuel_type_cat_electric  −$13,389
```

Every fuel type came out negative, which looks wrong on first glance. But the reference category (the dropped one, alphabetically first) is **diesel** — so every weight reads as "this fuel type is cheaper than a diesel." In the used market, diesel = heavy-duty pickups (F-250 Power Stroke, Ram 2500 Cummins), which are among the most expensive vehicles in the dataset. The negative signs are correct — diesel just happens to be a premium reference category.

**Teaching point for the slides**: dummy variable coefficients are always relative to the dropped reference category, and that choice changes how the numbers read.

---

## The full plain-LS arc (presentation summary)

1. Started with 4 numeric features. Got coefficients with **wrong signs** on year and mileage, and an absurd **+$18k** weight on `one_owner` — symptoms of multicollinearity and missing features.
2. Test R² was **−0.10** — worse than predicting the mean.
3. Added categoricals expecting improvement. Got **R² = −0.85** instead — adding more information made the model *worse*.
4. Diagnosed: **price outliers** including three $1,000,000,000 listings were destroying the regression. The bigger model had enough flexibility to chase them, distorting predictions.
5. Filtered to `$500 ≤ price ≤ $200,000`. Lost 0.16% of data.
6. After fix: clean baseline at **R² = 0.30**, full categorical model at **R² = 0.54**, RMSE dropped from $16k to $13k.
7. Every coefficient sign flipped to correct, magnitudes became realistic, and the manufacturer ordering matched intuition (Porsche premium, Mitsubishi discount).

This is the working "before / discovery / after" story for the plain-LS portion of the talk.

---

## Plain LS — extended with `model_cat` (filtered)

Added the `model_cat` column (specific car model, not just manufacturer) to the regression. Since `model_cat` has 666 levels including many very rare ones, the encoding was filtered: keep only models with ≥50 listings (434 models pass), and lump the rest (3,103 rows) into a single "Other" category. This avoided the singular-matrix crash we got on the raw 700-level version.

### Ladder of improvement (test set)

| Model | # features | Test R² | Test RMSE |
|---|---|---|---|
| Plain LS — 4 numeric only | 5 | 0.2956 | $16,147.86 |
| Plain LS — + 4 categoricals | 40 | 0.5441 | $12,991.07 |
| Plain LS — + filtered `model_cat` | 474 | **0.7751** | **$9,125.15** |

Each tier of features delivers real signal. Adding model info alone bumped R² by 0.23 and dropped RMSE by ~$4,000 per car.

### Challenge 6 — Plain LS is NOT overfitting, even at 474 features

The proposal anticipated that plain LS would overfit as the feature count grew, which would motivate Ridge and Lasso as fixes. The actual result:

- Train R² = 0.7754
- Test R² = 0.7751

**Identical to four decimal places.** Plain LS at 474 features is in the "tall and skinny" regime — 513k training rows ÷ 474 parameters ≈ 1,350 samples per coefficient, way more than enough to estimate them reliably. There is no overfitting to fix.

This changes the strategy for the Ridge/Lasso portion of the project. The textbook "plain LS overfits, Ridge saves it" story only appears when #features approaches or exceeds #training samples. With 641k rows, we'd need thousands of features (or to subsample the training set) to force that regime.

**Three options for making the Ridge/Lasso comparison land in the talk:**

1. **Push feature count much higher** — unfiltered model_cat dummies (~700), color one-hots, manufacturer × drivetrain interactions, etc. Goal: get to 1,500+ features so the train/test gap finally opens up.
2. **Subsample the training set** — run all three models on, say, 5,000 training rows. Plain LS will overfit hard at 474 features on that little data; Ridge will stay calm. Cleanest way to demonstrate the textbook story.
3. **Reframe Ridge/Lasso around interpretability, not overfitting** — Lasso's killer feature is automatic feature selection (zeros out columns that don't matter). Even if all three models tie on MSE, Lasso tells us *which* features carry signal — a different but compelling story.

Decision pending. For the presentation, option 2 (subsampled training set) is probably the highest-payoff move because it produces the canonical-looking feature-sweep curve.

---

## Plain LS — extended with colors

Added `exterior_color` and `interior_color` to the categorical feature list (498 total features). Used `np.linalg.lstsq` (SVD-based) instead of `np.linalg.solve` because adding model_cat alongside manufacturer creates structural collinearity (nested categoricals).

### Results

| Model | # features | Train R² | Test R² | Test RMSE |
|---|---|---|---|---|
| Plain LS — extended (with colors) | 498 | 0.7771 | 0.7771 | $9,085 |
| Plain LS — extended (no colors) | 474 | ~0.7754 | 0.7751 | $9,125 |

### Challenge 7 — Colors barely affect the model

Adding ~25 color dummies left R² essentially unchanged. This isn't a bug, it's *the finding*: in this dataset, color carries almost no information about price beyond what's already captured by manufacturer + model + drivetrain + fuel type. Two identical Civics — one black, one red — sell for nearly the same price.

**Why this matters for the project narrative:** the color dummies are the perfect setup for the Lasso "feature selection" demo. When we run Lasso, it should automatically drive most of those color coefficients to exactly zero, while Ridge will merely shrink them. Plain LS keeps all of them with small but nonzero weights. Three models, three different behaviors on the same useless features — that's the comparison story.

### Challenge 8 — One linearly dependent column

`lstsq` reported a rank deficit of 1 — meaning exactly one column in the design matrix is a linear combination of the others. This is a side effect of one-hot encoding nested categoricals (model nested within manufacturer). `np.linalg.solve` would have failed on this; `lstsq` handled it transparently by using SVD to find the minimum-norm least-squares solution.

**Talking point for the methodology slide:** "Naively one-hot encoding all categoricals creates structural collinearity when one category is nested within another (every model belongs to one manufacturer). We addressed this by switching from `np.linalg.solve` to `np.linalg.lstsq`, which solves the same minimization problem `min ‖y − Xw‖²` via SVD and handles rank-deficient design matrices."

---

## Ridge Regression — closed-form, swept across λ

Implemented from scratch in numpy as `ridge_solve(X, y, lam)` using the closed-form derivation:

**w = (XᵀX + λI)⁻¹ Xᵀy**

The only change from plain LS is the `+ λI` term inside the inverse. Since `XᵀX + λI` is strictly positive definite for any λ > 0, we could drop `lstsq` and use `np.linalg.solve` again — adding the diagonal boost fixes the rank-deficiency from earlier. The intercept (column 0) was excluded from penalization (set `I[0,0] = 0`) so the model's baseline price doesn't get artificially pulled to zero.

### Sweep results (λ from 0.001 to 1,000,000, on the 498-feature design)

| λ | Train R² | Test R² | Test RMSE | mean \|w\| |
|---|---|---|---|---|
| 0.001 | 0.7819 | **0.7833** | $8,956.67 | $19,517 |
| 1 | 0.7817 | 0.7831 | $8,961 | $11,600 |
| 100 | 0.7664 | 0.7681 | $9,266 | $5,933 |
| 10,000 | 0.5601 | 0.5653 | $12,686 | $718 |
| 1,000,000 | 0.3088 | 0.3138 | $15,939 | $18 |

**Best λ by test R²: 0.001** (i.e., the smallest value we tried). For comparison, plain LS got R² = 0.7833 — Ridge at λ=0.001 reproduces it to 4 decimals.

### Challenge 9 — Ridge doesn't improve test error here, and it's not Ridge's fault

Test R² is monotonically *decreasing* as λ grows. The classic "U-shape" of the bias-variance tradeoff is absent because plain LS isn't overfitting in this regime — 513k training rows for 498 parameters is the "tall, skinny" data regime where OLS is well-behaved. The whole point of L2 regularization is to fight overfitting; with no overfitting to fight, Ridge can only shrink weights, which trades bias for nothing.

### Challenge 10 — The shrinkage works exactly as the math predicts

Even though Ridge doesn't *win* on test error, it does what it's mathematically designed to do: drive coefficients smoothly toward zero as λ grows. From λ=0.001 to λ=1,000,000, mean |weight| dropped from $19,517 to $18 — a clean four-orders-of-magnitude decay. The log-log shrinkage plot is the headline visualization for the Ridge slide.

### Story for the slides

*"Ridge applies the L2 penalty exactly as derived: it shrinks weights toward zero as λ grows. In our dataset, regularization doesn't improve test performance because plain LS isn't overfitting — we have plenty of data per parameter. The optimal λ is essentially zero, meaning Ridge reduces to plain LS. But the shrinkage plot demonstrates that the penalty does what it's mathematically supposed to do, and it sets up the contrast with Lasso (next), which doesn't merely shrink — it zeros features entirely."*

This is an honest finding, not a failure: the project demonstrates that Ridge *works as designed* even when it doesn't *win* in a high-data setting.

---

## Lasso Regression — coordinate descent from scratch

Implemented Lasso from scratch using coordinate descent rather than using `sklearn.linear_model.LassoCV` as a black box. The algorithm cycles through coordinates one at a time, applying the closed-form 1D Lasso update for each:

```
ρⱼ = (1/n) Xⱼᵀ rⱼ                       (correlation with partial residual)
wⱼ = soft_threshold(ρⱼ, λ) / (Xⱼᵀ Xⱼ / n)
```

where `soft_threshold(z, λ) = sign(z) · max(|z| − λ, 0)` is the soft-thresholding operator — **the mechanic that makes Lasso produce exactly-zero coefficients.** When the correlation `|ρⱼ|` is below λ, the entire term zeroes out. This is the corner of the L1 penalty function manifesting in the optimization.

### Challenge 11 — Why Lasso has no closed-form solution

Plain LS and Ridge both have clean closed forms (`w = (XᵀX)⁻¹Xᵀy` and `w = (XᵀX + λI)⁻¹Xᵀy`) because their loss functions are smooth and differentiable everywhere. Lasso's loss includes `λΣ|wⱼ|`, and `|w|` is **not differentiable at zero** — the derivative is +1 on the right, −1 on the left, undefined exactly at 0. You can't set the gradient to zero and solve. The standard workaround is coordinate descent: optimize one coordinate at a time, where the 1D subproblem *does* have a closed form (the soft-thresholding update). Iterate until weights stop changing.

### Validation

Ran the from-scratch implementation on a 2,000-row subset against sklearn's Lasso with the same `α` parameter:

- Our implementation: 309/582 non-zero coefficients
- sklearn Lasso:      308/582 non-zero coefficients
- **Correlation between coefficient vectors: 0.9989**

Essentially identical results — confirming the implementation is mathematically correct. The single-coefficient difference is numerical precision near the soft-thresholding boundary.

### Talking point for the slides

*"We derived coordinate descent from the Lasso loss's subgradient structure and implemented it from scratch in 30 lines of numpy. To validate, we compared our coefficients with scikit-learn's optimized implementation — they agreed to a correlation of 0.999. For the cross-validated numbers on the full dataset, we used scikit-learn's optimized C++ implementation: same algorithm, ~100× faster than pure Python."*

This is the cleanest way to get the math credit AND the practical numbers without compromising either.

---

### Challenge 12 — Shipping the model to a phone with no server

We wanted a live demo: scan a QR code, open a website on your phone, pick a manufacturer/model, and get a fair-price estimate from our regression models. The naive approach is a Python backend (Flask/FastAPI) running the pickled models — but that means a server, cold starts, and hosting cost. We realized **all three of our linear models reduce to a single dot product**, so they can run entirely in the browser in JavaScript with no backend at all.

- **Plain LS / Ridge:** prediction is just `Xᵀw` on the raw 607-feature one-hot vector — the weight vectors `ls_weights` / `ridge_weights` are all the browser needs.
- **Lasso:** trickier, because Lasso is fit on *standardized* features (`(x − μ)/σ`), so its coefficients live in scaled units. Rather than ship the StandardScaler, we **folded the scaler into the weights** so the browser can still operate on raw inputs:

```
pred = b + Σ coefⱼ·(xⱼ − μⱼ)/σⱼ
     = [b − Σ coefⱼ·μⱼ/σⱼ]  +  Σ (coefⱼ/σⱼ)·xⱼ
        └──── new intercept ────┘   └─ effective weight ─┘
```

So we precompute `eff_wⱼ = coefⱼ/σⱼ` and a shifted intercept, and the browser does the same flat dot product as LS/Ridge. We verified the JavaScript predictions reproduce the Python/sklearn outputs **to within a few cents** (the only gap is rounding the exported weights to 6 decimals).

The dependent dropdown (manufacturer → models) needed a mapping that wasn't in the model bundle, so we built it from the cleaned CSV — the 557 distinct (manufacturer, model) pairs the models actually know — and restricted each manufacturer's list to those.

The whole model exports to a **55 KB JSON file**; the site is a static page deployed on Vercel, reachable by QR code with no running server.

### Talking point for the slides

*"Because least squares, Ridge, and Lasso are all linear, a trained model is nothing but a vector of weights — so we shipped the entire predictor as a 55 KB JSON file that runs in the browser, no server required. The one subtlety was Lasso: it's trained on standardized features, so we algebraically folded the standardization back into the weights. The JavaScript predictions match scikit-learn to the cent."*

This is a nice payoff of the linear-model math: the model literally *is* the closed-form solution `w = (XᵀX)⁻¹Xᵀy`, small enough to fit on a phone.

---

(Next: feature-pruning analysis on the full data, then the combined three-way comparison and the residual histograms for fair-price analysis.)
