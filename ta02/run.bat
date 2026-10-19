@echo off
REM Jalankan dari folder ta02 dengan venv aktif.
echo [1/2] Melatih model...
python train_model.py || (echo Gagal melatih model. Pastikan data\student-mat.csv ada. & pause & exit /b)
echo [2/2] Menjalankan web di http://127.0.0.1:5000
start http://127.0.0.1:5000
python app.py
