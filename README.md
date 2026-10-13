
## 📁 Project Directory Structure

```text
Capstone-1_MLOPS/
│
├── app/
│   └── main.py                  # FastAPI backend and prediction endpoint
│
├── src/
│   ├── train.py                 # Model training script
│   └── prediction.py            # Loads model and generates predictions
│
├── data/
│   ├── raw/                     # Original transaction dataset
│   └── processed/               # Cleaned and processed data
│
├── models/
│   └── Random_Forest.pkl        # Trained AML detection model
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and experiments
│
├── streamlit_app.py             # Streamlit frontend dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files excluded from Git
```
