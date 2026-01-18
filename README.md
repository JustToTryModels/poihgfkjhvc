# 🏢 Employee Churn Prediction 

<div align="center"> 
  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>
  
A machine learning project designed to predict employee attrition by analyzing key factors such as job satisfaction, workload, tenure, performance metrics, etc., helping organizations identify **at-risk** employees and implement proactive strategies to improve retention, boost productivity, and reduce turnover costs.


<br>

---

## 📋 Table of Contents

- [What is Employee Churn?](#-what-is-employee-churn)
- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Final Model Results](#-final-model-results)
- [Key Findings](#-key-findings)
- [Strategic Recommendations](#-strategic-recommendations)
- [Installation & Usage](#-installation--usage)

<br>

---

## ❓ What is Employee Churn?

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/employee-churn-PNG.png?raw=true" alt="Employee leaving with box" width="700"/>
</div>

**Employee churn** (also known as employee turnover or attrition) refers to employees leaving an organization, whether voluntarily or involuntarily. High churn rates can significantly impact a company through:

- 💰 Increased recruitment and training costs
- 📉 Loss of institutional knowledge and productivity
- 👥 Decreased team morale and engagement

This project uses machine learning to predict which employees are likely to leave, enabling HR teams to intervene early and improve retention strategies.

<br>

---

## 🎯 Project Overview

### Objective

Predict whether an employee will leave the company based on various factors such as job satisfaction, workload, tenure, and performance metrics.


<div align="center">

### 🛣️ Approach

| Component | Description |
|-----------|-------------|
| **Evaluation Scope** | Benchmarked **11 Classification Algorithms** |
| **Selected Model** | Random Forest Classifier |
| **Imbalance Handling** | Class weighting + Stratified K-Fold Cross-Validation |
| **Feature Selection** | Recursive Feature Elimination (RFE) & MDI |
| **Deployment** | Streamlit Web Application |

</div>

<br>

---

## 🚀 Demo

Try the live prediction model here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-turnover-predictor.streamlit.app/) 

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/Employee-Churn-GIF.gif?raw=true" alt="Employee" width="650" />
</div>

> Enter employee details and get instant predictions on their likelihood of leaving!

<br>

---

## 📁 Project Structure

```
employee-churn-prediction/
├── Data/
│   └── HR_comma_sep.csv                    # Dataset
├── Notebook/
│   └── Employee_Churn_Prediction.ipynb     # Jupyter notebook with model comparison
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

<br>

---

## 📊 Dataset

<div align="center">

### Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 11,991 (after removing duplicates) |
| **Stayed (Class 0)** | 10,000 (83.40%) |
| **Left (Class 1)** | 1,991 (16.60%) |

</div>

<br>

<div align="center">

### Features

| Feature | Description |
|---------|-------------|
| `satisfaction_level` | Employee's job satisfaction score (0-1) |
| `last_evaluation` | Score from most recent performance evaluation |
| `number_project` | Number of projects handled/completed |
| `average_monthly_hours` | Average hours worked per month |
| `time_spend_company` | Years spent at the company |
| `Work_accident` | Whether employee had a workplace accident (0/1) |
| `promotion_last_5years` | Whether promoted in last 5 years (0/1) |
| `Department` | Department where employee works |
| `salary` | Salary level (low/medium/high) |

</div>

<br>

---

## 🔬 Methodology

### 📊 1. Data Preparation
- **Train-Test Split:** 90% training, 10% testing.
- **Stratified splitting** to maintain class distribution.
- Test set locked for final evaluation only.

### ⚖️ 2. Imbalance Handling Strategy
- `class_weight='balanced'` in tree-based models.
- Stratified K-Fold Cross-Validation (k=5).
- Focus on both majority and minority class performance.

### 📌 3. Feature Selection
- **Method:** Recursive Feature Elimination (RFE) & Mean Decrease in Impurity (MDI).
- Iterative testing to find the minimum feature set required for maximum performance.

<br>

---

## ⚔️ Model Comparison & Selection

To ensure the most robust prediction capability, **11 different classification models** were evaluated. Feature selection techniques were tailored to each model where applicable:

- **Tree-based models** (e.g., Random Forest, XGBoost, LightGBM) used their own **internal feature importance metrics** (such as Mean Decrease in Impurity or Gain-based importance).
- **Other models** (e.g., Logistic Regression, SVC, KNN) were evaluated using **Recursive Feature Elimination (RFE)**.

All models were validated using **Stratified K-Fold Cross-Validation (k=5)** to ensure fair comparison and generalizability.

<br>

### 🚀 Performance Leaderboard (Test Set)

<div align="center">

| Model | Recall (Left) | Recall (Stayed) | Precision (Left) | Precision (Stayed) | Balanced Acc | Features Used |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** 🏆 | **0.9447** | **0.9980** | **0.9895** | **0.9891** | **0.9714** | **5** |
| Voting Classifier | 0.9447 | 0.9990 | 0.9947 | 0.9891 | 0.9719 | 9 |
| LightGBM | 0.9447 | 0.9950 | 0.9741 | 0.9891 | 0.9699 | 9 |
| XGB Classifier | 0.9397 | 0.9930 | 0.9639 | 0.9881 | 0.9664 | 17 |
| Gradient Boosting | 0.9497 | 0.9770 | 0.8915 | 0.9899 | 0.9634 | 8 |
| Decision Tree | 0.9548 | 0.9590 | 0.8225 | 0.9907 | 0.9569 | 2 |
| KNN Classifier | 0.9347 | 0.9860 | 0.9300 | 0.9870 | 0.9603 | 5 |
| SVC | 0.9296 | 0.9550 | 0.8043 | 0.9856 | 0.9423 | 8 |
| Logistic Regression | 0.8894 | 0.7842 | 0.4504 | 0.9727 | 0.8368 | 8 |

</div>

<br>

### 🌳 Why Random Forest?

Despite strong competition, **Random Forest** was selected for deployment based on the following criteria:

#### 1. Optimal Precision-Recall Balance
While the **Decision Tree** achieved slightly higher recall for leavers (95.48%), its precision dropped to 82.25%. This would result in a **17% false alarm rate**, causing HR to waste resources retaining employees who aren't actually leaving. Random Forest maintains a **98.95% precision**, ensuring that when the model flags a risk, it is almost certainly real.

#### 2. Superior to Complex Ensembles
The **Voting Classifier** achieved virtually identical results to Random Forest but required nearly double the features (9 vs 5) and the maintenance of multiple underlying models. The complexity cost was not justified by the negligible performance gain.

#### 3. Feature Efficiency & Interpretability
Random Forest achieved top-tier performance using only **5 key features**. Fewer features mean:
*   Less data collection overhead.
*   Lower risk of data drift.
*   Easier interpretation for stakeholders.

<br>

---

## 📈 Final Model Results


<div align="center">

### Selected Features (Top 5)

| Rank | Feature |
|------|---------|
| 1 | `satisfaction_level` |
| 2 | `time_spend_company` |
| 3 | `average_monthly_hours` |
| 4 | `number_project` |
| 5 | `last_evaluation` |

</div>

<br>

<div align="center">

### 💪 Performance Metrics (Random Forest)

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|--------|------------------|----------------|
| **Recall** | 0.9980 | 0.9447 |
| **Precision** | 0.9891 | 0.9895 |
| **F1-Score** | 0.9935 | 0.9666 |

</div>

<br>

<div align="center">

| Overall Metric | Score |
|----------------|-------|
| **Balanced Accuracy** | 0.9714 |
| **Overall Accuracy** | 0.9892 |

</div>

> ✅ **Model generalizes well - no overfitting detected!**

<br>

---

## 💡 Key Findings

### Summary

The organisation is facing a dual crisis of **burnout and stagnation**, systematically losing its most valuable employees. While low satisfaction (median 0.41 for leavers) is the immediate trigger, the root causes are structural: unsustainable workloads, a near-total lack of career progression, and compensation that fails to reward high effort.

### 🔑 1. The Core Predictors of Turnover

<div align="center">

| Metric | Employees Who Stayed | Employees Who Left | Key Insight |
|--------|----------------------|--------------------|-------------|
| Median Satisfaction | 0.69 | 0.41 | Low satisfaction is the common denominator for all departures |
| Median Evaluation | 0.71 | 0.79 | The organisation is systematically losing its highest-performing employees |
| Median Monthly Hours | 198 | 226 | Leavers are pushed significantly harder |
| Promoted (Last 5 Yrs) | 1.8% | 0.3% | Promotions are a powerful yet neglected retention tool |

</div>

<br>

### 📉 2. The Workload Crisis: A U-Shaped Curve of Risk

<div align="center">

| Risk Group | Hours/Month | Projects | Attrition Rate | Core Driver |
|------------|-------------|----------|----------------|-------------|
| 🔴 Extreme Burnout | 280–300+ | 6–7 | 62% – 100% | Unsustainable overload; 7 projects = 100% turnover |
| 🟢 The Safe Zone | 160–220 | 3–4 | 1% – 4% | Optimal balance with highest retention |
| 🟡 Under-Utilized | <160 | 2 | 30% – 54% | Boredom, disengagement, and poor role fit |

</div>

<br>

### 📆 3. Career Stagnation and the Tenure Cliff

<div align="center">

| Tenure (Years) | Attrition Rate | Observation |
|----------------|----------------|-------------|
| 2 | 1.1% | High initial retention |
| 3 | 17.9% | First major spike in departures |
| 5 | 45.6% | Peak attrition point; nearly half of this cohort leaves |
| 6+ | 0.0% | Extreme loyalty among those who survive the cliff |

</div>

**The Promotion Paradox:** While only 1.69% of the workforce has been promoted, these employees have an attrition rate of 3.9%, compared to 16.8% for unpromoted staff.

<br>

### 💰 4. Compensation and Departmental Risks

<div align="center">

| Salary Level | Attrition Rate | Workload Pattern | Key Insight |
|--------------|----------------|------------------|-------------|
| Low | 20.5% | More Hours: 223 vs. 197 | High performers are overworked and underpaid |
| Medium | 14.6% | More Hours: 229 vs. 199 | Moderate earners face similar burnout risks |
| High | 4.8% | Fewer Hours: 160 vs. 201 | Leavers are often under-utilized |

</div>

<br>

### 🚨 5. Critical Flight Risk Profiles

<div align="center">

| Cluster | Performance | Satisfaction | Workload | Primary Reason for Leaving |
|---------|-------------|--------------|----------|----------------------------|
| 🔥 The Burned Out | High (0.8–1.0) | Very Low (0.0–0.2) | Extreme Overtime | Exhaustion and lack of work-life balance |
| ⭐ The Poached Stars | High (0.8–1.0) | High (0.8–1.0) | High Overtime | Better external offers and lack of career growth |
| ❌ The Mismatched | Low (0.2–0.4) | Low (0.2–0.4) | Under-Utilized | Poor role fit or deep disengagement |

</div>

<br>

---

## 🎯 Strategic Recommendations

### 1. 🛑 Cap Workloads Immediately (Stop Burnout)

<div align="center">

| Action | Details |
|--------|---------|
| **The Rule** | No employee should be assigned more than 4 projects or work more than 220 hours/month |
| **The Fix** | Flag anyone working >220 hours. Redistribute their work to under-utilized staff |
| **Why** | 7 projects or 300+ hours guarantees 100% turnover |

</div>

<br>

### 2. 📈 Fix the "Mid-Career" Promotion Gap

<div align="center">

| Action | Details |
|--------|---------|
| **The Rule** | Implement a mandatory career review at the 3-year mark |
| **The Fix** | Create a clear promotion path. With only 1.69% of staff promoted in 5 years, the organisation is forcing its experienced staff to leave to advance their careers |
| **Why** | Promoted employees have ~4% churn rate vs. ~17% for non-promoted |

</div>

<br>

### 3. 💵 Pay High Performers Fairly

<div align="center">

| Action | Details |
|--------|---------|
| **The Rule** | Stop underpaying the hardest workers |
| **The Fix** | Audit Low and Medium salary brackets. Identify employees with High Evaluations (>0.8) and provide raises or bonuses |
| **Why** | The organisation is losing top talent because they work the most hours but receive the lowest pay |

</div>

<br>

### 4. 🎯 Engage the Under-Utilized

<div align="center">

| Action | Details |
|--------|---------|
| **The Rule** | Identify employees with <160 hours or only 2 projects |
| **The Fix** | Assign them more work, upskill them, or manage them out |
| **Why** | Boredom is driving nearly as much turnover as burnout |

</div>

<br>

### 5. 🏢 Target High-Risk Departments

<div align="center">

| Action | Details |
|--------|---------|
| **The Focus** | HR, Accounting, and Sales (highest churn) |
| **The Fix** | Conduct "Stay Interviews" in these specific departments to identify local stressors immediately |

</div>

<br>

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction.git
   cd Employee-Churn-Prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in the default browser at `http://localhost:8501`

<br>

---

## 🙏 Thank You

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true" alt="Thank You" width="400">
  
  If this project was helpful, please consider giving it a ⭐
</div>
