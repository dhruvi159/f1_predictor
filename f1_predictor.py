import os
import traceback
import fastf1
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def predict_final_race_winner(target_year=2024, target_gp="Abu Dhabi Grand Prix", train_last_n=6):
    """
    Train on historical races (excluding the target event) and predict the target event.
    Returns top-3 unique drivers with their teams and probabilities and the list of training events used.
    """

    # enable cache
    cache_dir = "f1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    def pick(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    def load_race_data(year, gp_name):
        """Load race + quali from cache/fastf1 and return a cleaned DataFrame row-per-driver."""
        try:
            sess_r = fastf1.get_session(year, gp_name, 'R')
            sess_r.load()
        except Exception as e:
            raise RuntimeError(f"race session load failed: {e}")

        # safe quali load
        quali_res = pd.DataFrame()
        try:
            sess_q = fastf1.get_session(year, gp_name, 'Q')
            sess_q.load()
            quali_res = getattr(sess_q, "results", pd.DataFrame())
        except Exception:
            quali_res = pd.DataFrame()

        race_res = getattr(sess_r, "results", None)
        if race_res is None or race_res.empty:
            raise RuntimeError("no race results")

        cols = list(race_res.columns)
        driver_col = pick(cols, ['Driver', 'DriverName', 'driver', 'Name', 'FullName'])
        team_col = pick(cols, ['Team', 'TeamName', 'Constructor', 'ConstructorName'])
        number_col = pick(cols, ['Number', 'No', 'DriverNumber', 'CarNumber'])
        grid_col = pick(cols, ['Grid', 'GridPosition', 'grid_pos'])
        pos_col = pick(cols, ['Position', 'Pos', 'finish_pos'])
        pts_col = pick(cols, ['Points', 'points', 'Pts'])

        if driver_col is None or team_col is None or pos_col is None:
            raise RuntimeError(f"essential columns missing: {cols}")

        df = pd.DataFrame({
            'driver_name': race_res[driver_col].astype(str),
            'team': race_res[team_col].astype(str),
            'grid_pos': race_res[grid_col] if grid_col in race_res.columns else 0,
            'finish_pos': race_res[pos_col],
            'points': race_res[pts_col] if pts_col in race_res.columns else 0
        })

        # quali gap
        try:
            if not quali_res.empty:
                # convert possible Q1/Q2/Q3 or best columns to seconds
                def to_sec_safe(x):
                    try:
                        return fastf1.utils.to_timedelta(x).total_seconds()
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return np.nan
                q_candidates = [c for c in quali_res.columns if c.upper() in ('Q1','Q2','Q3') or 'best' in c.lower()]
                if q_candidates:
                    secs = {}
                    for c in q_candidates:
                        try:
                            secs[c] = quali_res[c].apply(to_sec_safe)
                        except Exception:
                            secs[c] = pd.Series([np.nan]*len(quali_res))
                    secs_df = pd.DataFrame(secs)
                    if not secs_df.empty:
                        quali_res['best_quali_sec'] = secs_df.min(axis=1, skipna=True)
                        min_best = quali_res['best_quali_sec'].min(skipna=True)
                        quali_res['quali_gap'] = (quali_res['best_quali_sec'] - min_best).fillna(0)
                else:
                    quali_res['quali_gap'] = 0
        except Exception:
            quali_res['quali_gap'] = 0

        # align quali gap to drivers by name if possible
        if not quali_res.empty and 'Driver' in quali_res.columns:
            gap_map = dict(zip(quali_res['Driver'].astype(str), quali_res['quali_gap']))
            df['quali_gap'] = df['driver_name'].map(gap_map).fillna(0)
        else:
            df['quali_gap'] = 0

        # pit stops from laps (best-effort)
        try:
            laps = getattr(sess_r, "laps", pd.DataFrame())
            if not laps.empty and 'PitOutTime' in laps.columns and 'Driver' in laps.columns:
                pit_counts = laps[laps['PitOutTime'].notna()].groupby('Driver')['PitOutTime'].count().to_dict()
                df['pit_stop_count'] = df['driver_name'].map(pit_counts).fillna(0)
            else:
                df['pit_stop_count'] = 0
        except Exception:
            df['pit_stop_count'] = 0

        # weather features best-effort
        try:
            weather = sess_r.weather_data
            df['avg_temp'] = weather['TrackTemp'].mean() if 'TrackTemp' in weather.columns else 0
            df['rain_chance'] = weather['Rainfall'].mean() if 'Rainfall' in weather.columns else 0
        except Exception:
            df['avg_temp'] = 0
            df['rain_chance'] = 0

        # simple circuit type mapping (customize if needed)
        circuit_types = {
            'Monaco Grand Prix': 'street',
            'Singapore Grand Prix': 'street',
            'Italian Grand Prix': 'high-speed',
            'Belgian Grand Prix': 'high-speed',
            'Dutch Grand Prix': 'technical',
            'Abu Dhabi Grand Prix': 'technical',
        }
        df['circuit_type'] = circuit_types.get(gp_name, 'balanced')
        df['podium'] = df['finish_pos'].apply(lambda x: 1 if pd.notna(x) and int(x) in (1,2,3) else 0)
        df['event'] = f"{year} - {gp_name}"
        return df

    # build races_to_use from schedule, prefer cached schedule
    races_to_use = []
    for yr in [target_year, target_year - 1]:
        try:
            sched = fastf1.get_event_schedule(yr)
            if sched is None or getattr(sched, "empty", False):
                continue
            names = []
            if 'EventName' in sched.columns:
                names = sched['EventName'].dropna().tolist()
            elif 'OfficialEventName' in sched.columns:
                names = sched['OfficialEventName'].dropna().tolist()
            else:
                text_cols = sched.select_dtypes(include=['object']).columns
                if len(text_cols):
                    names = sched[text_cols[0]].dropna().tolist()
            for n in names:
                # skip the target event if it appears in schedule for the same year
                if not (yr == target_year and n == target_gp):
                    races_to_use.append((yr, n))
        except Exception:
            continue

    # keep last N historical races
    if not races_to_use:
        # fallback small hardcoded list
        races_to_use = [
            (target_year - 1, 'Abu Dhabi Grand Prix'),
            (target_year - 1, 'Brazilian Grand Prix'),
            (target_year - 1, 'United States Grand Prix'),
        ]
    races_to_use = races_to_use[-train_last_n:]

    # load historical races
    train_races = []
    for year, gp in races_to_use:
        try:
            print(f"[f1_predictor] Loading {year} - {gp}")
            train_races.append(load_race_data(year, gp))
        except Exception as e:
            print(f"[f1_predictor] Skipping {year} - {gp} - {e}")

    if not train_races:
        raise RuntimeError("No past race data could be loaded for training.")

    train_df = pd.concat(train_races, ignore_index=True)

    # aggregated features
    train_df['team_perf_season'] = train_df.groupby('team')['points'].transform('mean').fillna(0)
    train_df = train_df.sort_values(['driver_name', 'event']).reset_index(drop=True)
    train_df['driver_form_last5'] = (
        train_df.groupby('driver_name')['finish_pos']
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    ).fillna(train_df['finish_pos'].mean())

    train_df['avg_quali_gap'] = train_df.groupby('driver_name')['quali_gap'].transform('mean').fillna(0)
    train_df['pit_avg_by_circuit_type'] = train_df.groupby(['driver_name', 'circuit_type'])['pit_stop_count'].transform('mean').fillna(0)

    # features and encoding
    feat_base = ['grid_pos', 'quali_gap', 'team_perf_season', 'driver_form_last5',
                 'avg_quali_gap', 'pit_stop_count', 'pit_avg_by_circuit_type', 'avg_temp', 'rain_chance']
    X = train_df[feat_base].fillna(0)
    ct_dummies = pd.get_dummies(train_df['circuit_type'], prefix='ct')
    X = pd.concat([X, ct_dummies], axis=1).fillna(0)
    y = train_df['podium']

    # train model (avoid overfitting)
    # train calibrated model
    base_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    model.fit(X, y)

    # Build test set for the target event using entrants only (no target finish/points)
    try:
        sess_r_t = fastf1.get_session(target_year, target_gp, 'R')
        sess_r_t.load()
        entrants = getattr(sess_r_t, "results", pd.DataFrame()).copy()
    except Exception:
        entrants = pd.DataFrame()
    # if race session not available, try qualifying entrants
    if entrants.empty:
        try:
            sess_q_t = fastf1.get_session(target_year, target_gp, 'Q')
            sess_q_t.load()
            entrants = getattr(sess_q_t, "results", pd.DataFrame()).copy()
        except Exception:
            entrants = pd.DataFrame()

    # If no entrants, fallback to top historical drivers
    if entrants is None or entrants.empty:
        # predict on unique drivers in training data (aggregate by driver)
        preds = train_df.groupby(['driver_name', 'team'])['podium'].mean().reset_index()
        preds = preds.sort_values('podium', ascending=False).head(3)
        podium = []
        for i, r in preds.iterrows():
            podium.append({
                'position': len(podium) + 1,
                'driver_name': r['driver_name'],
                'team': r['team'],
                'podium_prob': float(r['podium'])
            })
        winner = podium[0] if podium else {'driver_name': None, 'team': None, 'podium_prob': 0.0}
        train_events = sorted(list(set(train_df['event'].tolist())), reverse=True)
        return {
            "winner_driver": str(winner["driver_name"]),
            "winner_team": winner["team"],
            "winner_prob": round(float(winner["podium_prob"]), 3),
            "podium": podium,
            "train_events": train_events
        }

    # normalize column names in entrants
    ecols = list(entrants.columns)
    dcol = pick(ecols, ['Driver', 'DriverName', 'driver', 'Name'])
    tcol = pick(ecols, ['Team', 'TeamName', 'Constructor'])
    gcol = pick(ecols, ['Grid', 'GridPosition', 'grid_pos'])
    # driver names
    entrants['driver_name'] = entrants[dcol].astype(str) if dcol else entrants.index.astype(str)
    entrants['team'] = entrants[tcol].astype(str) if tcol else entrants.get('Team', '').astype(str)
    entrants['grid_pos'] = entrants[gcol].fillna(0) if gcol else 0

    # map historical aggregates onto entrants
    team_perf_map = train_df.groupby('driver_name')['team_perf_season'].last().to_dict()
    form_map = train_df.groupby('driver_name')['driver_form_last5'].last().to_dict()
    avg_quali_map = train_df.groupby('driver_name')['avg_quali_gap'].last().to_dict()
    pit_map = train_df.groupby('driver_name')['pit_avg_by_circuit_type'].last().to_dict()

    test_df = pd.DataFrame()
    test_df['driver_name'] = entrants['driver_name'].astype(str)
    test_df['team'] = entrants['team'].astype(str)
    test_df['grid_pos'] = entrants['grid_pos'].astype(float)
    # attempt to derive quali_gap from qualifying session if available
    quali_gap = 0
    try:
        if 'sess_q_t' in locals() and not sess_q_t is None:
            qr = getattr(sess_q_t, "results", pd.DataFrame())
            if not qr.empty and 'Driver' in qr.columns:
                # compute best lap seconds
                def to_sec_safe(x):
                    try:
                        return fastf1.utils.to_timedelta(x).total_seconds()
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return np.nan
                qcols = [c for c in qr.columns if c.upper() in ('Q1','Q2','Q3') or 'best' in c.lower()]
                if qcols:
                    secs = pd.DataFrame({c: qr[c].apply(to_sec_safe) for c in qcols if c in qr.columns})
                    best = secs.min(axis=1, skipna=True)
                    min_best = best.min(skipna=True) if not best.isna().all() else 0.0
                    gap = (best - min_best).fillna(0.0)
                    gap_map = dict(zip(qr['Driver'].astype(str), gap))
                    test_df['quali_gap'] = test_df['driver_name'].map(gap_map).fillna(0.0)
                else:
                    test_df['quali_gap'] = 0.0
            else:
                test_df['quali_gap'] = 0.0
        else:
            test_df['quali_gap'] = 0.0
    except Exception:
        test_df['quali_gap'] = 0.0

    # historical aggregates mapped
    test_df['team_perf_season'] = test_df['driver_name'].map(team_perf_map).fillna(train_df['team_perf_season'].mean())
    test_df['driver_form_last5'] = test_df['driver_name'].map(form_map).fillna(train_df['driver_form_last5'].mean())
    test_df['avg_quali_gap'] = test_df['driver_name'].map(avg_quali_map).fillna(train_df['avg_quali_gap'].mean())
    test_df['pit_stop_count'] = 0  # unknown for target event
    test_df['pit_avg_by_circuit_type'] = test_df['driver_name'].map(pit_map).fillna(0)
    test_df['avg_temp'] = 0.0
    test_df['rain_chance'] = 0.0
    test_df['circuit_type'] = 'technical' if 'Abu Dhabi' in target_gp else 'balanced'

    # build test feature matrix consistent with training X
    test_X = test_df[feat_base].fillna(0)
    test_ct = pd.get_dummies(test_df['circuit_type'], prefix='ct')
    # ensure same dummy columns exist
    for c in X.columns:
        if c.startswith('ct_') and c not in test_ct.columns:
            test_ct[c] = 0
    test_X = pd.concat([test_X, test_ct.reindex(columns=[c for c in X.columns if c.startswith('ct_')]).fillna(0)], axis=1)
    # align columns in same order
    test_X = test_X.reindex(columns=X.columns, fill_value=0).fillna(0)

    # predict and aggregate
    proba = model.predict_proba(test_X)
    pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    test_df['podium_prob'] = proba[:, pos_idx]
    
    # Add normalized probabilities (sum to 100%)
    prob_sum = test_df['podium_prob'].sum()
    test_df['podium_prob_pct'] = (test_df['podium_prob'] / prob_sum * 100) if prob_sum > 0 else 0

    # Aggregate by driver (get max probability per driver)
    preds_by_driver = test_df.groupby(['driver_name', 'team']).agg({
        'podium_prob': 'max',
        'podium_prob_pct': 'max'
    }).reset_index()
    
    # Sort and get top 3
    preds_by_driver = preds_by_driver.sort_values('podium_prob', ascending=False).head(3)

    podium = []
    for i, row in preds_by_driver.iterrows():
        podium.append({
            'position': i + 1,
            'driver_name': row['driver_name'],
            'team': row['team'],
            'podium_prob': float(row['podium_prob']),
            'podium_prob_pct': float(row['podium_prob_pct'])
        })

    winner = podium[0] if podium else {'driver_name': None, 'team': None, 'podium_prob': 0.0}
    train_events = sorted(list(set(train_df['event'].tolist())), reverse=True)

    return {
        "winner_driver": str(winner["driver_name"]),
        "winner_team": winner["team"],
        "winner_prob": round(float(winner["podium_prob"]), 3),
        "podium": podium,
        "train_events": train_events
    }
