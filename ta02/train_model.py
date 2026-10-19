import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Muat data
df = pd.read_csv("data/student-mat.csv", sep=";")
df = df[["G2", "G3"]]
print("Jumlah data awal:", len(df))
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# 2. EDA: scatterplot (validasi asumsi linear)
plt.figure(figsize=(7, 5))
plt.scatter(df["G2"], df["G3"], alpha=0.6)
plt.xlabel("Nilai Periode 2 (G2)"); plt.ylabel("Nilai Akhir (G3)")
plt.title("Sebelum pembersihan: G2 vs G3")
plt.savefig("static/scatter_raw.png", dpi=150, bbox_inches="tight"); plt.close()

# 3. Preprocessing: baris dengan G3 = 0 adalah siswa yang tidak mengikuti ujian akhir
#    (putus/absen), bukan nilai sebenarnya -> dianggap outlier dan dibuang.
df_clean = df[(df["G3"] > 0) & (df["G2"] > 0)].copy()
print("Jumlah data setelah dibersihkan:", len(df_clean))
print("Korelasi G2-G3:", round(df_clean["G2"].corr(df_clean["G3"]), 3))

plt.figure(figsize=(7, 5))
plt.scatter(df_clean["G2"], df_clean["G3"], alpha=0.6)
plt.xlabel("Nilai Periode 2 (G2)"); plt.ylabel("Nilai Akhir (G3)")
plt.title("Setelah pembersihan: G2 vs G3")
plt.savefig("static/scatter_clean.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. Split & training
X = df_clean[["G2"]]
y = df_clean["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluasi
pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, pred)
rmse = float(np.sqrt(metrics.mean_squared_error(y_test, pred)))
mae = metrics.mean_absolute_error(y_test, pred)
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"Intercept (beta0): {model.intercept_:.4f}")
print(f"Koefisien G2 (beta1): {model.coef_[0]:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.7)
lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lim, lim, color="red")
plt.xlabel("G3 Aktual"); plt.ylabel("G3 Prediksi"); plt.title("Aktual vs Prediksi")
plt.savefig("static/actual_vs_pred.png", dpi=150, bbox_inches="tight"); plt.close()

# 6. Simpan model + metrik
joblib.dump(model, "model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"n_total": int(len(df)), "n_clean": int(len(df_clean)), "n_train": int(len(X_train)),
               "n_test": int(len(X_test)), "corr": round(float(df_clean["G2"].corr(df_clean["G3"])), 3),
               "r2": round(r2, 3), "rmse": round(rmse, 3), "mae": round(mae, 3),
               "intercept": round(float(model.intercept_), 3),
               "coef": round(float(model.coef_[0]), 3)}, f)
print("Model tersimpan: model.pkl")
