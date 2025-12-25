# 🏎️ Formula 1 Race Outcome Predictor

## Overview
This project implements a machine-learning pipeline to predict Formula 1 race outcomes using historical race, qualifying, telemetry, and weather data. The system emphasizes reproducibility, robust data handling, and feature-rich modeling by leveraging publicly available Formula 1 datasets accessed through the `fastf1` Python library.

The predictor combines individual driver performance, team strength, and race-specific context to model podium outcomes in Formula 1 races.

---

## Data Sources and Collection

### Primary Data Source
All data is retrieved programmatically using the **fastf1** library, which provides structured access to official Formula 1 timing and telemetry records.

To ensure reproducibility and speed, a local cache is enabled:

```python
fastf1.Cache.enable_cache("f1_cache")
```

## This ensures that:

-All API responses are cached locally

-Repeated experiments are faster

-Exact input datasets remain available for inspection

## Model Training

### Problem Formulation
The prediction task is framed as a **binary classification problem**.  
For each race entrant (driver), the model predicts whether the driver will finish on the **podium (positions 1–3)**.

- **Target variable:** `podium`
  - `1` → Podium finish
  - `0` → Non-podium finish

Each row in the training dataset represents a single driver participating in a specific Grand Prix.

---

### Model Choice
The primary model used is **LightGBM (Gradient Boosted Decision Trees)** due to:
- Strong performance on structured/tabular data
- Native handling of nonlinear feature interactions
- Robustness to missing values
- Fast training and inference

The model is implemented using the `lightgbm` Python library and trained via a scikit-learn compatible API.

---

### Input Features
The model is trained on a combination of engineered features across three dimensions:

**Driver Performance**
- `quali_gap`
- `driver_form_last5`
- `avg_quali_gap`

**Team / Constructor Performance**
- `team_perf_season`
- `pit_avg_by_circuit_type`

**Race Context**
- `grid_pos`
- `circuit_type`
- `avg_temp`
- `rain_chance`

> Note: `pit_stop_count` is used only during training and excluded from inference since pit stops are not known before a race.

Categorical features (e.g., `circuit_type`) are encoded prior to training.

---

### Training Procedure
- Training data is constructed using the most recent `train_last_n` races
- Feature aggregation is performed **before** splitting data to avoid leakage
- The dataset is split chronologically, ensuring that future races are never used to predict past events
- Class imbalance (podium vs non-podium finishes) is handled implicitly by the boosting algorithm

---

## Model Evaluation

### Evaluation Strategy
Model performance is evaluated on a **held-out race event**, simulating real-world prediction where only historical data is available.

Predictions are generated for all drivers in the target race and compared against actual race results.

---

### Evaluation Metrics
The following metrics are used to assess performance:

- **Accuracy**  
  Overall correctness of predictions

- **Precision**  
  Fraction of predicted podium finishes that were correct

- **Recall**  
  Fraction of actual podium finishers correctly identified

- **F1-Score**  
  Harmonic mean of precision and recall

These metrics provide a balanced view of performance, especially under class imbalance.

---

### Baseline Comparison
Model performance is implicitly compared against simple baselines such as:
- Predicting podium purely based on grid position
- Predicting podium based on historical averages only

The trained model consistently outperforms naive baselines by incorporating multi-factor context.

---

### Interpretation and Feature Importance
LightGBM provides feature importance scores, allowing insight into model behavior.

Commonly influential features include:
- `quali_gap`
- `grid_pos`
- `driver_form_last5`
- `team_perf_season`

This aligns with domain knowledge, reinforcing the model’s validity.

---

## Inference and Usage

At inference time (pre-race prediction):
- Only features available **before the race** are used
- The model outputs a podium probability for each driver
- Drivers are ranked based on predicted probability

This allows flexible downstream usage such as:
- Podium likelihood rankings
- Top-3 driver predictions
- Comparative analysis across drivers and teams

---

## Limitations
- Weather forecasts may differ from actual race conditions
- Strategy variables (e.g., pit stops) cannot be fully known before the race
- Driver retirements and safety cars are not explicitly modeled

Despite these limitations, the model captures the dominant performance signals affecting race outcomes.

---

## Future Work
Potential improvements include:
- Multi-class prediction (P1, P2, P3)
- Lap-level time-series models
- Incorporation of practice session performance
- Ensemble models for improved stability
- Calibration of probability outputs

---
