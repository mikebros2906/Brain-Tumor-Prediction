@echo off
cd /d "E:\Jupyter notebooks\ML Projects\Brain Tumor Detection Prediction"
call .venv\Scripts\activate
streamlit run app/app.py
pause
