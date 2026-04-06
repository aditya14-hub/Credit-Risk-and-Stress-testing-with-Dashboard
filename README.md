🏦 Lender’s Club — Credit Risk & Stress Testing Dashboard

An end-to-end Credit Risk Modeling & Stress Testing System built using real-world lending data, simulating how financial institutions assess risk, estimate losses, and evaluate portfolio resilience under adverse scenarios.

🚀 Overview

This project replicates a mini credit risk engine used in banks and fintech companies. It combines:

📊 Predictive Modeling (PD timation) ⚖️ Risk Segmentation 📉 Stress Testinges 💰 Expected Credit Loss (ECL) estimation 🖥️ Interactive dashboard (Streamlit) 🧠 Key Features 🔹 1. Credit Risk Modeling Built classification models: Logistic Regression Random Forest XGBoost Evaluated using: ROC-AUC KS Statistic Precision / Recall / F1

👉 Final model selected based on performance and business relevance

🔹 2. Feature Engineering (Risk-Oriented) WOE (Weight of Evidence) transformation Information Value (IV) based feature selection Handling class imbalance Leakage-aware modeling 🔹 3. Risk Segmentation Converted model probabilities into: Low Risk Medium Risk High Risk

👉 Enables decision-making for lending policies

🔹 4. Stress Testing (What-if Scenarios) Simulated adverse economic conditions using: PD scaling (stress multiplier) Observed: Risk migration across buckets Increase in default probabilities 🔹 5. Expected Credit Loss (ECL) Implemented simplified IFRS 9 framework: 𝐸𝐶𝐿=𝑃𝐷×𝐿𝐺𝐷×𝐸𝐴𝐷

Adjustable parameters: PD (via stress multiplier) LGD (Loss Given Default) EAD (Exposure at Default)

👉 Provides portfolio-level loss estimation

🔹 6. Interactive Streamlit Dashboard Real-time borrower simulation Adjustable risk factors via sliders Scenario analysis with: Normal vs Stressed Loss Additional Loss Loss Ratio

👉 Designed like a real internal risk tool

📊 Sample Insights Higher credit utilization and interest rates significantly increase default probability Stress scenarios (e.g., 1.5× PD) lead to substantial increases in portfolio loss Risk segmentation helps identify high-risk borrower clusters for targeted actions 🧩 Tech Stack Python (Pandas, NumPy, Scikit-learn) XGBoost Streamlit Matplotlib / Seaborn 📁 Dataset Lending Club dataset (via Kaggle) ~30,000 loans with borrower and financial features ⚠️ Assumptions & Limitations Simplified ECL calculation (constant LGD & EAD) Stress applied via PD scaling (not macroeconomic modeling) No time-series or real-time deployment

👉 Designed for learning + demonstration of risk concepts

🎯 Use Cases Credit Risk Analysis Risk Modeling Demonstration Fintech / Banking Interviews Portfolio Risk Simulation
