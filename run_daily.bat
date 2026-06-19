@echo off
cd /d "c:\Users\kim anh\OneDrive\Documents\GitHub\AI-Prediction-model-"
".venv\Scripts\python.exe" daily_update.py >> "logs\task_scheduler.log" 2>&1
