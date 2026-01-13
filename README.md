# 🛡️ Financial Fraud Detection in Online Payments

Real-time fraud detection pipeline built with XGBoost & Random Forest. Features rigorous class imbalance handling, custom feature engineering for "account draining" patterns, and SHAP interpretability. Achieved 1.00 Recall on imbalanced financial data to secure mobile money transactions.

This notebook contains a robust machine learning pipeline designed to identify fraudulent mobile money transactions in real-time. By prioritizing high recall, this system effectively detects financial anomalies to protect vulnerable populations who rely on mobile transfer services.

---

## 📂 Repository Structure

This repository contains the end-to-end development of the fraud detection engine, from exploratory data analysis to the final deployable pipeline.

| File | Description |
| --- | --- |
| **`Online_Payments_Fraud_Detection.ipynb`** | **✨ The Core Pipeline.** Contains the complete workflow: Data ingestion, extensive EDA, feature engineering, model training (XGBoost, Random Forest, LightGBM), and SHAP interpretability analysis. |
| **`fraud_detection_pipeline.pkl`** | **Deployment Artifact.** The serialized model pipeline (including preprocessing steps) ready for API integration. |
| **`requirements.txt`** | **Dependencies.** Lists all necessary Python libraries to reproduce the environment. |

---

## 🛠️ Methodology & Engineering

I implemented a rigorous Machine learning lifecycle with a specific focus on handling extreme class imbalance and ensuring model interpretability.

### 1. Data Preprocessing & Engineering

I focused on reducing noise and isolating the mechanisms of fraud.

* **Filtration:** I filtered the dataset to include only `TRANSFER` and `CASH_OUT` transaction types. I found fraud exclusively occurs in these channels; including others introduces noise.
* **Feature Engineering:** I created domain-specific features to capture criminal behavior:
* **`balance_error`:** Calculates discrepancies betIen the transaction amount and the balance change. This became the strongest predictor.
* **`is_Account_Drain`:** Flags transactions where the amount equals the exact account balance.
* **`is_Max_Limit`:** Flags transactions hitting the system's hard transfer cap.


* **Scaling:** I applied `StandardScaler` within the pipeline. While tree-based models (XGBoost) function without scaling, I included it to ensure robustness for potential future model swaps (e.g., Logistic Regression) and to stabilize interpretability outputs.

### 2. Handling Class Imbalance

The dataset presents an extreme imbalance with only 0.13% fraud cases. I rejected standard accuracy metrics and applied a multi-layered mitigation strategy.

* **Cost-Sensitive Learning (Iighting):** I altered the loss function rather than the data.
* **XGBoost:** I calculated `scale_pos_Iight` to penalize false negatives.
* **Scikit-Learn:** I used `class_Iight='balanced'`. This adjusts Iights inversely proportional to class frequency.
* **Stratified Splitting:** I used `stratify=y` during the train-test split. This enforces the preservation of the 0.13% fraud ratio in both training and testing sets.
* **Metric Selection:** I optimized for **Recall** (finding the fraud) and **F1-Score**. I avoided Accuracy, as a model predicting "Safe" for every case would achieve 99.87% accuracy but fail its objective.
* **Why No SMOTE:** I avoided Synthetic Minority Over-sampling Technique (SMOTE) for two reasons:
* **Efficiency:** Generating synthetic neighbors for millions of rows increases memory usage inefficiently.
* **Data Integrity:** Adding synthetic points to an already synthetic simulation (PaySim) risks introducing artifacts.



### 3. Model Architecture & Tuning

I benchmarked multiple algorithms to identify the optimal balance of speed and performance.

* **Baseline:** XGBoost Classifier.
* **Challengers:** Random Forest, LightGBM, Logistic Regression.
* **Tuning:** I used `RandomizedSearchCV` for a stability check. Since the baseline Random Forest achieved an F1-score of 0.9985, extensive grid searching offered diminishing returns. The tuning process validated the stability of our parameters.

---

## 🧠 Interpretability & Insights

I utilized **SHAP (SHapley Additive exPlanations)** to eliminate the "Black Box" nature of the model and explain individual predictions.

* **The "Account Draining" Pattern:** EDA and SHAP analysis confirmed that fraudsters typically liquidate the entire account balance in a single step.
* **The Balance Error:** The `balance_error` feature proved critical. It identifies mathematical inconsistencies in the ledger that occur when fraudsters manipulate the backend system.
* **Time Sensitivity:** The `step` feature (representing time) shoId high importance, indicating specific hours carry higher risk profiles.
-----

## 💻 How to Use the Notebooks

To reproduce the results or train the model yourself, follow these steps:

### Prerequisites

  * A Google account (if using Colab) or a local Python environment with Jupyter.
  * A **Kaggle API Token** (`kaggle.json`).

### Step-by-Step Guide

1.  **Clone the Repository**

    ```bash
    git clone [[https://github.com/Shaguns26/PlantDiseaseDetection.git](https://github.com/Shaguns26/Fraud-Detection-in-Online-Payments.git)](https://github.com/Shaguns26/Fraud-Detection-in-Online-Payments.git)
    cd Fraud-Detection-in-Online-Payments
    ```

2.  **Open the Scaled Notebook**
    Open `Online_Payments_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab.

3.  **Setup Kaggle Credentials**
    The notebook requires a `kaggle.json` file to download the datasets.

      * **In Colab:** The notebook has a cell to upload the file directly.
      * **Local:** Ensure `kaggle.json` is in your `~/.kaggle/` directory.

4.  **Run the Cells**
    Execute the blocks sequentially. The notebook will:

      * ⬇️ Download and unzip the datasets.
      * 🔄 Merge and filter the data into a clean structure.
      * 🧠 Train the models (Baseline, Optimized, and ResNet).
      * 📊 Generate evaluation metrics (Accuracy, RMSE) and plots.

-----

## 📊 Results

The final Random Forest model demonstrated exceptional performance on the test set.

* **Recall (Fraud Class):** **1.00**
* I successfully identified 100% of the fraudulent transactions in the test set.


* **Precision (Fraud Class):** **0.91**
* I accepted a marginally loIr precision to ensure maximum recall. This trade-off ensures client safety; a false positive is an inconvenience, but a false negative is a financial loss.



---

## 👤 Author

* **Shagun Sharma** - *Machine Learning Engineer*

**Graduate Student, Duke University - Fuqua School of Business**

  * [GitHub Profile](https://github.com/Shaguns26)

