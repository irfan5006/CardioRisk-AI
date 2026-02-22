<p align="center">
  <img src="assets/heart_icon.png" alt="CardioVision AI Logo" width="140"/>
</p>

<h1 align="center">🫀 CardioVision AI</h1>

<p align="center">
  <strong>AI-Powered Heart Disease Risk Prediction System</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-KNN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  A professional, medical-themed web application that leverages a <strong>K-Nearest Neighbors (KNN)</strong> machine learning model to predict heart disease risk based on patient clinical data. Built with <strong>Streamlit</strong> for an interactive and visually polished user experience.
</p>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📊 Dataset](#-dataset)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [🧠 Model Details](#-model-details)
- [📸 Screenshots](#-screenshots)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📬 Contact](#-contact)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-Time Prediction** | Instantly predicts heart disease risk from 11 clinical input features |
| 📊 **Probability Scores** | Displays both Low Risk and High Risk probability percentages |
| 🎨 **Medical-Themed UI** | Soft pastel gradients, glassmorphism cards, and polished typography |
| 📱 **Responsive Layout** | Two-column input layout that adapts to different screen sizes |
| ⚡ **Fast Inference** | Pre-trained KNN model with StandardScaler for instant predictions |
| ⚕️ **Disclaimer Notice** | Built-in medical disclaimer for responsible AI usage |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **ML Library:** [Scikit-Learn](https://scikit-learn.org/) (K-Nearest Neighbors)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Visualization:** Custom HTML/CSS with Google Fonts (Inter & Poppins)

---

## 📊 Dataset

This project uses the **Heart Failure Prediction Dataset** containing **918 records** with 11 clinical features:

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | M = Male, F = Female |
| `ChestPainType` | Categorical | ATA, NAP, TA, ASY |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dL) |
| `FastingBS` | Binary | 1 if fasting blood sugar > 120 mg/dL, else 0 |
| `RestingECG` | Categorical | Normal, ST, LVH |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Y = Yes, N = No |
| `Oldpeak` | Numeric | ST depression induced by exercise |
| `ST_Slope` | Categorical | Up, Flat, Down |

> **Target Variable:** `HeartDisease` — 1 (Heart Disease), 0 (Normal)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/heart_disease_ml_project.git
   cd heart_disease_ml_project
   ```

2. **Create a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run src/app.py
   ```
   Or use the launcher script:
   ```bash
   python run.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

---

## 📁 Project Structure

```
heart_disease_ml_project/
│
├── 📂 assets/                    # Static assets & media
│   └── heart_icon.png            # Application logo / icon
│
├── 📂 data/                      # Datasets
│   └── heart.csv                 # Heart disease dataset (918 records)
│
├── 📂 models/                    # Trained ML model artifacts
│   ├── KNN_heart.pkl             # Trained KNN classifier
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── columns.pkl               # Expected feature column names
│
├── 📂 notebooks/                 # Jupyter notebooks for EDA & training
│   └── Heart.ipynb               # Full EDA, training & evaluation pipeline
│
├── 📂 src/                       # Application source code
│   ├── __init__.py               # Package initializer
│   └── app.py                    # Streamlit web application (entry point)
│
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # Project documentation (this file)
├── requirements.txt              # Python dependencies
└── run.py                        # Convenience launcher script
```

---

## 🧠 Model Details

| Aspect | Detail |
|---|---|
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Preprocessing** | One-Hot Encoding for categoricals, StandardScaler for numerics |
| **Training Pipeline** | Data Cleaning → Feature Engineering → Scaling → Model Training → Evaluation |
| **Serialization** | Joblib (`.pkl` files in `models/` directory) |

### Workflow

```
data/heart.csv
    │
    ▼
Exploratory Data Analysis (notebooks/Heart.ipynb)
    │
    ▼
Feature Engineering (One-Hot Encoding)
    │
    ▼
Data Scaling (StandardScaler)
    │
    ▼
KNN Model Training & Evaluation
    │
    ▼
Model Export → models/ (KNN_heart.pkl, scaler.pkl, columns.pkl)
    │
    ▼
Streamlit Web App → src/app.py
```

---

## 📸 Screenshots

> *Run the app locally and take screenshots to add here.*

<!-- Uncomment and update paths after adding screenshots:
<p align="center">
  <img src="assets/screenshots/home.png" alt="Home Page" width="80%"/>
  <br><em>Home Page — Patient Clinical Details Input</em>
</p>

<p align="center">
  <img src="assets/screenshots/result.png" alt="Prediction Result" width="80%"/>
  <br><em>Risk Analysis Results with Probability Scores</em>
</p>
-->

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Muhammad Irfan**

- 🌐 GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

<p align="center">
  Developed with ❤️ by <strong>Muhammad Irfan</strong> &nbsp;|&nbsp; Machine Learning Project
</p>
