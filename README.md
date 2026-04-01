# 6G Smart City IDS — Intrusion Detection System

## Overview
This project was developed as part of the PIDEV – 3rd Year Engineering Program at **Esprit School of Engineering** (Academic Year 2025–2026).

It implements a unified Machine Learning-based Intrusion Detection System (IDS) for 6G Smart City networks, covering four traffic slices: mMTC, URLLC, eMBB, and TON_IoT. The pipeline includes data quality assessment, EDA, feature selection, multi-model training, and a three-stage attack detection & classification output.

## Features
- Multi-dataset support: mMTC, URLLC, eMBB, TON_IoT
- Data quality reports (missing values, duplicates, outliers)
- Class imbalance handling with SMOTE
- Feature selection per dataset (justified per feature importance)
- Multiple ML models: Random Forest, Gradient Boosting, Extra Trees, Logistic Regression, MLP
- Model comparison summary with metrics (accuracy, precision, recall, F1)
- Feature importance validation
- Three-stage final decision pipeline: detection → classification → output

## Tech Stack
### Frontend
- N/A (Notebook-based project)

### Backend / ML
- Python 3
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## Architecture
Three-stage pipeline:
1. **Detection** — Binary classification (normal vs. attack)
2. **Classification** — Multi-class attack type identification
3. **Output** — Final decision report per network slice

## Contributors
- [Yahyaoui Anas] — https://github.com/anasYaha
- [Karray Ahmed] — https://github.com/ahmed-karray
- [Klouz Rami] — https://github.com/ramiklouz
- [Kheder Tasnim] — https://github.com/tasnim-dev-ux
- [Belkadhi Mustapha Aziz] — https://github.com/WorKenX306
- [Bendhaw Zahra] — https://github.com/z-a-h-r-a


## Academic Context
Developed at **Esprit School of Engineering – Tunisia**
PIDEV – 4 DATA | 2025–2026

## Getting Started
```bash
# Clone the repository

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Open the notebook
jupyter notebook 6G_IDS_final.ipynb
```

> **Note:** Datasets (mMTC, URLLC, eMBB, TON_IoT CSVs) must be placed in a `/data/` folder or mounted via Google Drive as configured in the notebook.

## Acknowledgments
- Esprit School of Engineering – Tunisia
- Dataset sources: [cite your dataset origins, e.g. TON_IoT benchmark]
