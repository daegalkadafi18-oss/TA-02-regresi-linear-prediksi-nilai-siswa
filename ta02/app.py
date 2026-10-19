import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

try:
    model = joblib.load("model.pkl")
    with open("metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    raise SystemExit("model.pkl belum ada. Jalankan dulu: python train_model.py")


def kategori(nilai):
    if nilai < 10:
        return "Perlu Perhatian", "danger", "Prediksi di bawah batas kelulusan (10). Siswa sebaiknya mendapat pendampingan."
    if nilai < 14:
        return "Cukup", "warning", "Prediksi lulus, namun masih ada ruang untuk ditingkatkan."
    if nilai < 17:
        return "Baik", "success", "Prediksi nilai akhir tergolong baik."
    return "Sangat Baik", "primary", "Prediksi nilai akhir sangat tinggi."


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(result=None, error=None, g2="", kat=None, m=METRICS)
    if request.method == "POST":
        ctx["g2"] = request.form.get("g2", "").strip().replace(",", ".")
        try:
            value = float(ctx["g2"])
            if not 0 <= value <= 20:
                raise ValueError
            pred = model.predict(pd.DataFrame({"G2": [value]}))[0]
            pred = round(max(0, min(20, pred)), 2)   # skala nilai 0-20
            ctx["result"] = pred
            ctx["kat"] = kategori(pred)
            ctx["selisih"] = round(pred - value, 2)
        except ValueError:
            ctx["error"] = "Masukkan angka antara 0 dan 20."
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
