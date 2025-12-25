from flask import Flask, render_template, jsonify
from f1_predictor import predict_final_race_winner
import numpy as np
import os
import requests
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    try:
        result = predict_final_race_winner()
        print(result)
        # If it's a pandas DataFrame
        if hasattr(result, "replace"):
            result = result.replace([np.nan, np.inf, -np.inf], None)
            result = result.to_dict(orient="records")

        # If it's a dict or list with NaN values
        elif isinstance(result, (dict, list)):
            import math
            def clean_nans(obj):
                if isinstance(obj, dict):
                    return {k: clean_nans(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nans(x) for x in obj]
                elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                else:
                    return obj
            result = clean_nans(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/diag')
def diag():
    info = {}
    # Python packages
    try:
        import fastf1
        info['fastf1'] = fastf1.__version__
    except Exception as e:
        info['fastf1'] = f"import error: {e}"

    # Internet check (simple HTTP)
    try:
        r = requests.get("https://ergast.com/api/f1/current/last/results.json", timeout=8)
        info['internet_status'] = r.status_code
    except Exception as e:
        info['internet_status'] = f"error: {e}"

    # FastF1 schedule test + cache write test
    try:
        fastf1.Cache.enable_cache("f1_cache")
        sched = fastf1.get_event_schedule(2024)
        info['schedule_rows'] = len(sched) if sched is not None else None
        info['schedule_cols'] = list(sched.columns) if sched is not None else None
    except Exception as e:
        info['schedule_error'] = str(e)
        info['schedule_traceback'] = traceback.format_exc()

    # test cache folder writable
    try:
        test_path = os.path.join("f1_cache", "diag_test.txt")
        os.makedirs("f1_cache", exist_ok=True)
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        info['cache_writable'] = True
    except Exception as e:
        info['cache_writable'] = f"error: {e}"

    return jsonify(info)

# Optional: prevent favicon.ico 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == "__main__":
    app.run(debug=True)
