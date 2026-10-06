# Omnix™ ML Studio 1.2

*Machine learning, predictive analytics and model interpretation for biomedical and omics data*

A standalone, leakage-safe machine-learning application with five modules: **classification,
regression, clustering, anomaly detection and time-series forecasting**. Each module follows the
same numbered workflow, modelled on PyCaret's (setup → compare → create/tune → analyze →
finalize → predict). It is fully independent of the Omnix™ platform. The modelling engine
(`src/ml_studio/core`) has **no Streamlit dependency** and can be imported into any Python
application.

> Software tests verify that the code does what it is designed to do. They are **not**
> scientific validation of any model built with it.

---

## Changelog

**v1.2.1: two speed fixes found while building the Jupyter edition**

* Logistic regression, Gaussian process, Theil–Sen and linear regression no longer get `n_jobs` > 1: for them
  scikit-learn starts worker *processes* and memory-maps the data on every fit, which on the TCGA matrix made
  tuning logistic regression take many minutes and left temporary folders in shared memory. Their folds now run
  in parallel threads (100-iteration tuning on TCGA: about 15 s).
* Create & tune always uses the fold cache (it was skipped when more than one CPU core was selected).
* The memory limit for a model was set from the memory free at that instant, which macOS and Linux report as
  a few hundred MB when much RAM is used as cache; a model needing 0.5 GB (AdaBoost on TCGA) was then stopped at
  0.3 GB. The limit is now the larger of 80% of free memory and half of the computer's RAM (set `OMNIX_MEMORY_GB`
  to choose it yourself).
* Predicting new samples with 10,000+ features builds the feature table in one step (no fragmentation warnings).

**v1.2.0: fast and crash-proof on real omics data**

Tested end to end on a TCGA tumour/normal RNA-seq matrix (485 samples × 10,962 genes, 444 tumour / 41 normal,
40 patients with paired samples) on a 2-core, 7 GB server:

| | before | v1.2 |
|---|---|---|
| Compare all 20 classifiers, all genes reaching the models | > 25 min (sum of single-model runs; CatBoost alone > 15 min); blocks the page | 7.9 min, in the background (peak 2.1 GB) |
| Compare all 20, top-1,000 in-fold pre-filter (the default for > 2,000 features) | not measured | **43 s** (peak 0.9 GB) |
| Data-quality checks / each click on the Data step | 11.7 s / 2.6 s | 2.3 s / 0.2 s |
| Permutation importance, 1,000 genes × 10 repeats | ~ 16–18 min (extrapolated from 100 genes) | 16–89 s (model-dependent) |
| Gaussian Process on 10,962 genes | AUC 0.500 (learned nothing) | AUC 1.000 |

- **Fold cache**: preprocessing (filters, imputation, scaling, in-fold feature selection) is fitted once per
  training fold and reused by every model and every tuning setting. Each fold's preprocessing still sees only
  that fold's training rows; scores are identical to per-model pipelines (tested to 1e-9, including SMOTE and
  in-fold selection), and the leakage tests now run through this path.
- **All CPU cores**, inside the protected worker processes: native threads for LightGBM / XGBoost / CatBoost /
  forests / KNN, parallel folds for the rest. `OMNIX_MAX_THREADS` caps it on shared servers.
- **Background jobs**: Compare (all modules), random-search tuning, nested CV and permutation importance run off
  the page. Clicking, switching steps or modules no longer cancels them; progress, finished models and a
  **Stop** button stay visible, and a line under the module bar shows what is running.
- **Time limit per model** now also skips a model early when its first fold projects past the limit.
- **CatBoost on wide data** (> 500 features): 32 split borders and ~100 features per split level, 50× faster
  with the same AUC; defaults unchanged on narrow data. **Gaussian Process**: RBF length scale starts at √p.
- **Random search** reuses the fold cache (same settings as scikit-learn's `RandomizedSearchCV`, same seed).
- **Permutation importance**: transform once, shuffle model-input columns, batch the predictions (identical
  results; KNN imputation keeps the exact slow path).
- **Fixed**: the gene RUNX3 was guessed as a *batch* column ("run"), causing a false batch-confounding alert
  and dropping it from the features. Role guesses now match whole words. The model picker no longer loses its
  selection after navigating mid-run; the patient column added from TCGA barcodes is applied reliably.
- **New**: *All* preset in the algorithm picker; `psutil` pinned (the protected runner needs it; a fresh install
  failed without it).

**v1.1.0: analysis controls, figure export, clearer workflow**
- **Analyze**: choose the comparison. Binary: pick the positive class. Multiclass: one-vs-rest for chosen classes,
  or class A vs class B (only those samples; score = P(A) / (P(A) + P(B))). Threshold, calibration and gain/lift now
  work for multiclass through that choice. Confusion matrix as row %, column % or counts. ROC legends show AUC.
- **Feature importance for genes only**: rank *all features*, *genes / omics* or *clinical variables*, and show the
  top 10, 15, 20, 25, 30, 40, 50, 75 or 100. Genes and clinical variables are separated by the clinical covariates
  marked in the Data step, or else by naming pattern (800 genes / 14 clinical on the example). The same filter is in
  Explain (SHAP, permutation, model-specific).
- **All plots**: one switch renders every diagnostic for the chosen model (all five modules), and **Download all
  plots** saves them as a ZIP.
- **Figure download, every chart**: PNG, JPEG, WebP (150–1,200 DPI, 3.5–14 in wide, DPI written into the PNG), SVG
  and PDF (vector), and self-contained interactive HTML. Static export uses Kaleido with Chromium, which the
  Dockerfile and `packages.txt` (Streamlit Cloud) now install. The app finds Chrome/Chromium automatically.
- **Training set** slider in Setup (60–100%) with live counts: *Training set* N and *Independent holdout (test set)*
  M, in samples or patients.
- **Buttons and steps** redesigned: every button is outlined or filled, the selected tab is filled, the step bar shows
  *Completed / Current step / Ready / Locked*, and the selected plot in a list is highlighted.
- **Automatic next step** after confirming Data, Setup and the final model; after any step with results, a
  *Continue to …* bar appears at the top.
- **Leakage scan, multiclass**: a numeric feature that alone separates one class is now flagged (e.g. `pack_years`
  is exactly 0 for every never-smoker, so it determines `smoking_status = Never`). ID columns are no longer offered
  as outcomes.
- **Preview (not yet in the app)**: `preview/start_preview.py`, a Start screen that reads an uploaded table,
  summarises it and suggests the analysis type with reasons and cautions (`core/advisor.py`).

**v1.0.0: redesigned workspace and all PyCaret-equivalent algorithms**
- New layout: a **module bar** (5 modules, each with its own colour and its own independent
  session state) and a **numbered step bar**. Locked steps open only once the earlier steps
  are complete; finished steps carry a check. Results sit on the left, and a **grouped
  settings panel** with one Run button sits on the right. The tabs and sidebar are gone.
- **Algorithm picker** grouped by family, with *Default / All fast / Clear* presets and badges
  (e.g. "no probabilities", "slow", "batch only"). It replaces the long multiselect.
- **Hyperparameters**: *Defaults*, *Custom* (edit any tunable value), or *Random search* with
  the optimisation metric and its search space shown. Every created or tuned model is
  cross-validated **on the same folds as the comparison** and shows a per-fold table.
- **Analyze** gallery (from out-of-fold predictions): ROC, precision–recall, confusion matrix,
  threshold, calibration, gain/lift, class report, learning curve, feature importance and
  hyperparameters. Regression adds residuals, prediction error and Q–Q.
- **Predict & deploy**: finalize on the development set or on all samples, then download a
  **signed model bundle**. New samples are predicted with compatibility checks and
  out-of-distribution flags.
- **New modules**: clustering, anomaly detection and time series (details in section 3).
- Models added to match PyCaret: classification now has 20 algorithms and regression 26,
  including XGBoost and CatBoost. Models without probability estimates (Ridge, linear SVM)
  are handled correctly: they get no AUC or log loss where those are undefined, and nested CV
  then selects by balanced accuracy.
- Iterations: random search, nested-CV tuning and clustering/anomaly stability now default to **100** (ranges 10–500, 10–300 and 20–500), with the number of fits and an estimated run time shown next to each control.
- Removed: the old page-based UI (`dashboard`, `*_page.py`, `common.py`).

Earlier releases: v0.4.0 feature selection and explainability; v0.3.1 hotfix (log-transform
guard, all-model failure message); v0.3.0 Phases 1–3.

## 1. Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12 (not 3.13+: pmdarima/numba)
pip install -r requirements.txt
streamlit run app.py                                       # http://localhost:8501
```

Developers:

```bash
pip install -r requirements-dev.txt
pytest                      # 291 tests, ~8 min (includes end-to-end UI runs of every module)
pytest -m "not ui"          # engine only, ~2 min
ruff check .
```

**Streamlit Community Cloud:** under *Advanced settings*, choose **Python 3.11**. Newer Python
versions fail to build `pmdarima`/`numba`.

**Figure export:** PNG/JPEG/WebP/SVG/PDF export needs Chrome or Chromium on the server (Docker image and
`packages.txt` include it). Without it, the HTML download and the chart's camera button (PNG at 4×) still work.

**Preview of the Start screen:** `streamlit run preview/start_preview.py`.

**CPU and memory:** the app uses every core it is given and runs one model per worker process with a memory
limit. On a shared or small server set `OMNIX_MAX_THREADS`. For data with more than ~2,000 features keep the
in-fold univariate pre-filter (on by default): it is the largest speed and memory saving.

**Model bundles across restarts:** set the environment variable `OMNIX_MODEL_KEY` to a long
random secret. Without it, bundles are signed with a per-process key and can only be reloaded
while the server keeps running.

### Docker

```bash
docker build -t omnix-ml-studio .
docker run -p 8501:8501 -e OMNIX_MODEL_KEY=change-me omnix-ml-studio
```

### Render

1. Push this folder to a Git repository.
2. Render → **New → Web Service** → connect the repo → **Runtime: Docker** (it uses the `Dockerfile`).
3. Add `OMNIX_MODEL_KEY` as a secret environment variable. No start command is needed; the
   container reads `$PORT`.
4. Instance: at least 1 GB RAM (2 GB recommended for nested CV, more than 5,000 features, or
   CatBoost and the slow forecasters). Keep **Parallel workers = 1** on small instances
   (Setup → Validation design → Advanced).

---

## 2. Workflow

Every module uses the same layout: a step bar, results on the left, and settings with a Run
button on the right.

| Step | Classification / regression | Clustering | Anomaly detection | Time series |
|---|---|---|---|---|
| 1 Data | Upload or example; roles (outcome, positive class, ID, patient/group, batch, time, covariates); 17 quality checks; explore | Features, sample ID, optional **reference label** (evaluation only); scale and missing-value checks | same as clustering (reference = 0/1 anomaly flag) | Value and date columns; regular frequency inferred, missing periods inserted and filled |
| 2 Setup | Imputation, transform (scope + offset guard), scaling, filters, in-fold feature selection, imbalance, CV design; **holdout sealed**; optional selection-stability study | Imputation, transform, scaling, filters, optional PCA | same + contamination | Horizon, seasonal period, rolling origins, **sealed last horizon**, interval coverage; ADF/KPSS and STL strengths |
| 3 Compare | All algorithms on identical folds; dummy baseline; best value per metric highlighted; per-fold spread | Internal criteria (silhouette, Calinski–Harabasz, Davies–Bouldin) and, with a reference, ARI/AMI/homogeneity/completeness/V-measure | Reference metrics (ROC-AUC, PR-AUC, precision@n) or, without one, consensus overlap; score-rank and flag-overlap agreement | MASE, RMSSE, MAE, RMSE, MAPE, sMAPE across rolling origins; "beats seasonal naive" |
| 4 Create & tune | Defaults / custom / random search; per-fold table; choose the final model | Parameter editor; **k scan** across all criteria (+ inertia, BIC) | Parameter editor + contamination | Parameter editor; CV forecasts per origin |
| 5 Analyze | Plot gallery (see changelog) | PCA / t-SNE maps, sizes, silhouette, versus reference | Score distribution, ranked scores, maps, top anomalies | STL decomposition, ACF/PACF, CV forecasts, CV residuals + Ljung–Box |
| 6 Validate | Sealed holdout (bootstrap CIs, ledger) · nested CV · external cohort (compatibility, shift) | **Clusterwise bootstrap Jaccard** stability, ARI between refits | **Flag stability** across refits | Holdout forecast once, interval coverage |
| 7 Explain | SHAP (summary, importance, dependence, single sample), permutation, model-specific, PDP/ICE | Cluster profiles (Kruskal–Wallis / χ², BH-FDR; labelled descriptive) | Robust-z deviation profile for a flagged sample | (none) |
| 8 Deploy | Finalize (development set or all data), signed bundle, predict new data with OOD flags | Assignments, bundle, assign new samples | Scores, bundle, score new samples | Refit on the whole series, forecast ahead, bundle |

## 3. Algorithms

| Module | Count | Algorithms |
|---|---|---|
| Classification | 20 | Logistic Regression, L1 Logistic Regression, Ridge Classifier, linear SVM (SGD), LDA, QDA, Naive Bayes, KNN, SVM (RBF), Gaussian Process, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, LightGBM, XGBoost, CatBoost, MLP, Dummy |
| Regression | 26 | Linear, Ridge, Lasso, Elastic Net, LARS, Lasso-LARS, OMP, Bayesian Ridge, ARD, Passive Aggressive, RANSAC, Theil–Sen, Huber, KNN, SVR, Kernel Ridge, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, LightGBM, XGBoost, CatBoost, MLP, Dummy |
| Clustering | 9 (+1) | K-Means, Gaussian Mixture, Agglomerative, BIRCH, Spectral, DBSCAN, OPTICS, Mean Shift, Affinity Propagation (+ K-Modes when `kmodes` is installed) |
| Anomaly detection | 12 | Isolation Forest, KNN, LOF, COF, CBLOF, HBOS, PCA, MCD, One-Class SVM, ABOD, SOD, SOS (PyOD) |
| Time series | 31 | Naive, Grand mean, Seasonal naive, Polynomial trend, ARIMA, Auto ARIMA, Holt–Winters, ETS, Theta, STLF, Croston, BATS, TBATS, and 18 reduction models (Linear, Elastic Net, Ridge, Lasso, Lasso-LARS, Bayesian Ridge, Huber, OMP, Passive Aggressive, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, LightGBM, XGBoost, CatBoost), each with conditional deseasonalising and detrending (sktime); Prophet when installed |

**Why the engine is not PyCaret itself.** PyCaret's `setup()`/`compare_models()` cannot run
nested CV, cannot guarantee that every learned step is refit inside each fold, and supports
only Python ≤ 3.11. The app therefore reproduces PyCaret's workflow and algorithm coverage on
scikit-learn, imbalanced-learn, PyOD and sktime, so it can enforce the safeguards below.

## 4. Scientific safeguards (enforced in code and tested)

| Risk | Safeguard | Test |
|---|---|---|
| Holdout influences development | Split before modelling; development functions take only `train_idx`; holdout evaluations logged in a `HoldoutLedger`, with a warning on reuse | `test_holdout_rows_never_seen_during_development`, `test_holdout_values_do_not_affect_learned_preprocessing` |
| Feature selection on all data | `SelectKBest` and every filter sit inside the pipeline, so they are refit per training fold | `test_feature_selection_inside_folds_removes_optimism` (pure noise: naive selection AUC > 0.85, engine ≈ 0.5), `test_selector_fitted_only_on_training_rows` |
| Oversampling leaks into evaluation | imbalanced-learn pipeline, so samplers run only in `fit` | `test_resampling_only_in_training_folds` (training folds balanced; every test row real and unmodified) |
| Repeated measures split across folds | Grouped CV, grouped holdout and repeated grouped CV; an ungrouped strategy is **refused** when a group column is set | `test_groups_never_cross_folds`, `test_grouped_cv_removes_patient_fingerprint_leakage` (row-wise AUC > 0.9 vs grouped ≈ 0.5) |
| Tuning sees outer test fold | Nested CV: tuning/selection run on outer-train rows only | `test_tuning_never_sees_outer_test_fold` (traces every fit call) |
| External data refit | Saved pipeline applied as-is; shift diagnostics use a separate model | `test_external_validation_does_not_refit` |
| Silent pseudocounts | log transforms raise on values ≤ 0 unless an explicit offset is chosen | `test_log_transform_refuses_nonpositive_without_offset` |
| Correlated units in CIs | Bootstrap resamples groups when present | `test_cluster_bootstrap_is_wider_than_naive` |

Metric conventions: for binary outcomes the positive class is user-chosen and encoded as 1.
For multiclass outcomes, ROC-AUC is one-vs-rest macro and precision/recall/F1 are macro
averages. MAPE is reported only when every |y| > 0. The leaderboard flags models that do
not beat the dummy baseline.

### Unsupervised and forecasting safeguards

| Risk | Safeguard | Test |
|---|---|---|
| Accepting clusters that any algorithm would produce | Bootstrap Jaccard per cluster with Hennig's thresholds; post-selection caveat on profiles | `test_stability_distinguishes_real_from_forced_structure`, `test_profiles_and_assignment` |
| Choosing k by one criterion | k scan shows every criterion; the tests document where they disagree | `test_compare_and_kscan_choose_true_k` |
| Using the reference label to fit | Reference kept outside the feature matrix; used only for evaluation | `test_reference_label_is_never_a_feature` |
| Scoring new data with batch-only detectors | COF, SOD and SOS are marked transductive: they cannot be bundled, `score_new` refuses them, and stability uses only subsample members | `test_transductive_detectors_cannot_score_new_data`, `test_transductive_stability_runs` |
| Look-ahead in forecasting | Expanding-window origins; each fit sees only the past; last horizon sealed | `test_split_is_temporal_and_sealed`, `test_cv_never_sees_the_future` |
| Scale-dependent forecast scores | MASE/RMSSE scaled by the in-sample seasonal-naive error | `test_mase_definition`, `test_seasonal_models_beat_naive_and_mase_reference` |
| Executing an uploaded pickle | Bundles are verified with HMAC-SHA256 **before** unpickling; foreign, edited or extra-file archives are refused | `test_tampered_or_foreign_bundles_are_refused` |

## 5. Test execution summary (actual results)

Run on Python 3.11.15 in a clean virtual environment installed from `requirements.txt`:

```
291 passed in 498.20s     (pip check: no broken requirements; ruff: all checks passed)
```

| File | Covers |
|---|---|
| test_data_validation.py | Loading, oversize/empty files, integrity checks, leakage scan, imbalance, batch confounding, audited exclusions |
| test_preprocessing.py | Every imputation × transform × scaling combination, log safety, filters, feature names |
| test_classification.py | All 20 classifiers (binary and multiclass), metrics, imbalance modes, models without probabilities in holdout and nested CV |
| test_regression.py | All 26 regressors, MAPE rule |
| test_splitting.py, test_leakage.py, test_nested_cv.py | Group integrity, the six mandatory leakage scenarios, nested CV, holdout ledger, cluster bootstrap, external validation |
| test_feature_selection.py, test_explainability.py | 9 selection methods and stability; SHAP additivity, permutation, PDP direction |
| test_clustering.py | Every algorithm, k scan, internal/external metrics, bootstrap stability, profiles, assignment of new samples |
| test_anomaly.py | Every detector, score orientation, label metrics, transductive refusal, flag stability, deviation profile |
| test_timeseries.py | Series preparation (irregular dates, gaps), no look-ahead, MASE/RMSSE, every forecaster, intervals, holdout, final forecast, diagnostics |
| test_model_persistence.py, test_workflow_core.py | Bundle round-trip and tamper refusal; create-model fold identity; learning curve; new-sample prediction with OOD flags; clustering/anomaly/forecast bundles |
| test_ui_regressions.py | Invalid log design blocked; all-model failure explained; locked steps; per-module state isolation |
| test_ui_workflow.py | The full 8-step (7 for time series) workflow of every module in Streamlit AppTest: binary, multiclass, regression, clustering, anomaly, time series |

Browser screenshots of every module were also reviewed during development.

## 6. Architecture

```
app.py                         Streamlit entry point (module bar → step rail → workspace)
src/ml_studio/
  core/                        ← import this; no Streamlit
    config.py, data_validation.py, preprocessing.py, splitting.py
    classification.py          20 classifiers + search spaces, grouped by family
    regression.py              26 regressors
    clustering.py              9 algorithms, k scan, bootstrap stability, profiles, assignment
    anomaly.py                 12 PyOD detectors, agreement, flag stability, deviation profile
    timeseries.py              31 sktime forecasters, rolling-origin CV, MASE/RMSSE, diagnostics
    metrics.py, model_comparison.py (compare_models, create_model), optimization.py (tune)
    validation.py              fit_final, evaluate_holdout, nested_cv, external_validate, cv_predictions, learning curve
    feature_selection.py, explainability.py
    model_manager.py           signed bundles, training profile, OOD detector, predict_new
    synthetic.py               ground-truth datasets for tests and examples
  visualization/               Plotly figure builders (no Streamlit)
  ui/
    shell.py                   design tokens, module bar, step rail, panel/card/run helpers
    components.py              model picker, parameter editor, downloads
    supervised.py + sup_data.py, sup_model.py, sup_eval.py
    unsupervised.py + unsup_common.py, clustering_ui.py, anomaly_ui.py, timeseries_ui.py
  utils/session_utils.py       per-module session state with downstream invalidation
tests/                         pytest suite (+ ui_flow_helpers.py for AppTest)
examples/                      synthetic datasets (each module has its own)
```

### Using the engine from another application

```python
import sys; sys.path.insert(0, "path/to/omnix_ml_studio/src")
from ml_studio.core.config import StudyDesign, PreprocessConfig, ValidationConfig
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.splitting import holdout_split
from ml_studio.core.model_comparison import compare_models, create_model
from ml_studio.core.validation import fit_final, evaluate_holdout

design = StudyDesign(target="group", task="binary", features=feature_cols, group="patient_id")
data = prepare_xy(df, design)
split = holdout_split(data.y, "binary", 0.2, data.groups)
val = ValidationConfig(cv_strategy="stratified_group_kfold")
prep = PreprocessConfig(transform="log2", log_offset=1.0, univariate_k=50)
board = compare_models(data, "binary", ["dummy", "lr", "rf", "xgboost"], prep, val, train_idx=split.train_idx)
pipe = fit_final(data, split.train_idx, "binary", "lr", prep, val)
result = evaluate_holdout(pipe, data, split.test_idx, "binary")
```

## 7. Known limitations

**Scientific**

- Nested CV estimates the performance of the *selection procedure*, not of the single final
  model. Report it as such.
- Bootstrap CIs on a small holdout are wide and themselves unstable. Below about 50
  independent units, nested CV is usually more informative than a holdout.
- A tuned CV score is optimistic (best of N); use the holdout or nested CV for an honest
  estimate. "Finalize on all samples" produces a model that has not itself been evaluated.
- Thresholds read off the Analyze threshold plot must be confirmed on the holdout.
- External validation reports the original model only; recalibration is not implemented. Batch
  information is used only for diagnostics; there is no batch correction, by design.
- Stability measures consistency, not validity. Predictive is not the same as significant.
- SHAP values are in the model's output space and are not comparable across model types.
  Model-agnostic SHAP is capped at 300 model features.
- **Clustering:** internal criteria favour compact, convex clusters and often disagree (on the
  default example, silhouette prefers k = 2 while BIC finds the true k = 3). Clustering always
  returns clusters, even in structureless data, so use stability. Cluster profiles are
  descriptive: their p-values are not valid tests because the clusters were built from the same
  features. Algorithms without `predict` (hierarchical, spectral, DBSCAN, OPTICS) assign new
  samples to the nearest training sample, which is an approximation and is labelled as one.
  DBSCAN's default `eps` often labels everything noise after standard scaling; tune `eps`.
- **Anomaly detection:** contamination sets the threshold; it is an assumption, not an
  estimate. COF, SOD and SOS are transductive and cannot score new data. The deviation profile
  shows single-feature extremes only; a detector may flag unusual combinations.
- **Time series:** univariate only (no exogenous regressors). Prediction intervals are available
  only for models that support them in sktime 0.26 (naive family, ARIMA, Auto ARIMA, ETS,
  Theta, BATS, TBATS). With a short horizon, interval coverage is very imprecise. CV residual
  tests pool across origins and are approximate.

**Software**

- State is per session and in memory; nothing is cached across users. Bundles are the way to
  keep a model.
- Exact reproducibility holds with a fixed seed and `n_jobs=1`; results can differ slightly
  with more workers or other library versions.
- There is no PDF report yet. Every table has CSV/Excel download, and charts have Plotly's
  PNG button.

*Research use only. Not a medical device.*
