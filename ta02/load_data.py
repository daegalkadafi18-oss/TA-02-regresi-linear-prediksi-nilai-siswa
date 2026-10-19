# Bukti akses & validasi data (TP-02) -> screenshot kode + output ini
import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")   # file dari Kaggle memakai pemisah titik koma

print("--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Nama Kolom ---")
print(df.columns)
