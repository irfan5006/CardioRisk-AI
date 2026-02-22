"""
CardioVision AI — Heart Disease Risk Prediction
Run this script to launch the Streamlit web application.

Usage:
    streamlit run src/app.py
    OR
    python run.py
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
