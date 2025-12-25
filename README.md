🏎️ Formula 1 Race Outcome Predictor
Overview

This project implements a machine-learning pipeline to predict Formula 1 race outcomes using historical race, qualifying, telemetry, and weather data. The system emphasizes reproducibility, robust data handling, and feature-rich modeling, leveraging publicly available Formula 1 datasets accessed via the fastf1 Python library.

The predictor is designed to model driver performance by combining individual form, team strength, and race-specific context, making it suitable for experimentation, research, and data-driven motorsport analysis.

Data Sources and Collection
Primary Data Source

All raw data is retrieved programmatically using the fastf1 library, which provides structured access to official Formula 1 timing and telemetry data.

To ensure reproducibility and efficient experimentation, the project enables a persistent local cache:

fastf1.Cache.enable_cache("f1_cache")


This guarantees that:

Every call to fastf1.get_event_schedule(...) and
fastf1.get_session(..., 'R'/'Q')
is cached locally

Inputs are versioned and can be inspected later

Re-running experiments does not depend on live API responses

Extracted Data Sources

For each Grand Prix, the pipeline extracts three principal categories of data:

1. Race & Qualifying Results

Structured result tables containing:

Driver name

Constructor (team)

Grid position

Finishing position

Championship points

2. Lap-Level Telemetry & Metadata

Lap timing data

Pit stop inference using PitOutTime

Used to compute pit stop counts per driver

3. Session Weather Data

Track temperature time series

Rain indicators

Aggregated into:

avg_temp

rain_chance

Robust Data Loader

The core loader function:

load_race_data(year, gp_name)


Includes defensive logic to handle real-world data inconsistencies:

Dynamically searches for alternative column names
(e.g., Driver, DriverName, Constructor, Grid, Position, Points)

Uses safe fallbacks when qualifying or race sessions are missing

Employs explicit try/except blocks to avoid pipeline failures

This prioritizes robustness over brittle assumptions, ensuring the model can train across multiple seasons and race formats.

Derived Features (Per-Race)

Immediately after loading, the pipeline constructs analysis-ready features:

quali_gap
Best qualifying lap (seconds) relative to pole position

pit_stop_count
Number of pit-out events detected per driver

circuit_type
Categorized as:

street

high-speed

technical

balanced

podium (target label)
Binary classification:

1 → Finished in positions 1–3

0 → Finished outside podium

Aggregated Predictors Across Races

To capture longer-term performance trends, the system computes rolling and historical statistics across multiple events:

team_perf_season
Average championship points scored by each team in the season

driver_form_last5
Rolling average of a driver’s finishing positions over the last 5 races

avg_quali_gap
Historical average qualifying gap for each driver

pit_avg_by_circuit_type
Average pit stop frequency per driver on similar circuit layouts

These features are joined onto the target event’s driver list during test-set construction.

Feature Engineering Pipeline

Once cleaned, raw race data is transformed through a structured feature engineering pipeline that captures three dimensions of Formula 1 performance.

1️⃣ Individual Driver Performance

quali_gap – One-lap pace relative to the fastest qualifier

driver_form_last5 – Recent race momentum

avg_quali_gap – Long-term qualifying consistency

2️⃣ Team / Constructor Performance

team_perf_season – Car competitiveness and team efficiency

pit_avg_by_circuit_type – Strategy tendencies on similar tracks

3️⃣ Race-Specific Context

grid_pos – Starting position

pit_stop_count – Strategy complexity (training only)

circuit_type – Track layout classification

avg_temp, rain_chance – Weather-driven race dynamics

Additional transformations may include:

Polynomial feature generation to capture nonlinear interactions

Time-series aggregations for performance trends

Training Data Selection

Training races are selected using the most recent train_last_n events

A small hardcoded fallback list is used if schedule retrieval fails

This approach balances realism with dataset completeness

Reproducibility & Dependencies

The repository explicitly documents:

Required Python packages:

fastf1

pandas

scikit-learn

lightgbm

Cached data directory (f1_cache/)

Deterministic feature construction logic

These choices ensure experiments can be reproduced and audited reliably.

Summary

This project demonstrates a complete, real-world sports analytics pipeline:

Reliable data ingestion from public motorsport records

Defensive engineering against missing or inconsistent data

Domain-aware feature construction grounded in Formula 1 race dynamics

Reproducible experimentation through caching and explicit dependencies

It provides a strong foundation for predictive modeling, performance analysis, and further research in motorsport data science.
