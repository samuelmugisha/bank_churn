

# Bank Churn Prediction

Predicting customer churn to help banks identify at-risk customers and implement proactive retention strategies.
Using Neural Networks
---

## Overview

**What problem are you solving?**
This project aims to predict customer churn in a bank, identifying customers who are likely to leave in the next 6 months.

**Why is it important?**
Customer churn is a critical issue for businesses like banks, impacting revenue and growth. By predicting churn, banks can proactively engage with at-risk customers, improve satisfaction, and reduce customer attrition.

**Who benefits from the solution?**
Bank management and marketing teams will benefit by gaining insights into customer behavior and enabling targeted retention campaigns.

---

## Business Problem

**The real-world challenge:**
The core challenge is understanding which aspects of banking services influence a customer's decision to leave. Customer churn leads to significant revenue loss and increased acquisition costs for new customers.

**Project objectives:**
To build a neural network-based classifier that can accurately determine whether a customer will leave the bank or not in the next 6 months.

**Key questions to answer:**
- Which customers are most likely to churn?
- What are the primary factors contributing to customer churn?
- How accurately can a model predict customer churn, especially identifying true churners?

---

## Dataset

### Description
- **Number of records:** 10,000
- **Number of features:** 14 initially, reduced to 11 after preprocessing.
- **Target variable:** `Exited` (0 = No Churn, 1 = Churn)

### Key Variables
| Feature           | Description                                                               |
|-------------------|---------------------------------------------------------------------------|
| CustomerId        | Unique ID assigned to each customer (dropped during preprocessing)      |
| Surname           | Last name of the customer (dropped during preprocessing)                  |
| CreditScore       | Credit history of the customer                                            |
| Geography         | Customer’s location (France, Spain, Germany)                              |
| Gender            | Gender of the customer                                                    |
| Age               | Age of the customer                                                       |
| Tenure            | Number of years the customer has been with the bank                       |
| NumOfProducts     | Number of products purchased through the bank                             |
| Balance           | Account balance                                                           |
| HasCrCard         | Whether the customer has a credit card (binary)                           |
| IsActiveMember    | Whether the customer is an active member of the bank (binary)             |
| EstimatedSalary   | Estimated salary of the customer                                          |
| Exited            | Target variable: whether the customer left the bank (0=No, 1=Yes)         |

---

## Project Workflow

### 1. Data Collection
- The dataset `bank.csv` was loaded from Google Drive into a pandas DataFrame.

### 2. Data Cleaning
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`.
- Checked for missing values (none found).
- Checked for unique values and duplicated rows (none found).

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Histograms and box plots were used to understand the distribution of numerical features (`CreditScore`, `Age`, `Balance`, `EstimatedSalary`). Bar plots were used for categorical features (`Exited`, `Geography`, `Gender`, `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`).
- **Bivariate Analysis:** 
    - Correlation heatmap was generated for numerical features to identify relationships.
    - Stacked bar plots were used to analyze the relationship between categorical features and the target variable `Exited`.
    - Box plots were used to examine the relationship between numerical features and `Exited`.

**Major findings from EDA:**
- The target variable `Exited` is imbalanced, with approximately 20% churn.
- `Age`, `Balance`, `Geography` (Germany), and `IsActiveMember` (inactive members) showed a notable relationship with customer churn.
- `CreditScore`, `Tenure`, `EstimatedSalary`, and `HasCrCard` showed less direct influence on churn.

### 4. Feature Engineering
- **Encoding methods:** Categorical features (`Geography`, `Gender`) were converted into numerical representations using one-hot encoding (`pd.get_dummies`).
- **Scaling techniques:** Numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`) were standardized using `StandardScaler` to bring them to a similar scale.
- **Data Splitting:** The data was split into training, validation, and test sets (80/10/10 split).

### 5. Model Development
Neural Network models were built and evaluated:
- **Model 0:** Neural Network with SGD Optimizer.
- **Model 1:** Neural Network with Adam Optimizer.
- **Model 2:** Neural Network with Adam Optimizer and Dropout.
- **Model 3:** Neural Network with SMOTE (Oversampling) and SGD Optimizer.
- **Model 4:** Neural Network with SMOTE (Oversampling) and Adam Optimizer.
- **Model 5:** Neural Network with SMOTE (Oversampling), Adam Optimizer, and Dropout.
- **Model 6:** Neural Network with Random Undersampling, Adam Optimizer, and Dropout.

All models used a `Sequential` Keras model with `Dense` layers and `sigmoid` activation for the output layer. `Recall` was chosen as the primary evaluation metric due to the higher cost of False Negatives (missing actual churners).

### 6. Model Evaluation
- **Evaluation metrics used:** Recall, Precision, F1-score, and Accuracy were used, with a primary focus on Recall for the churn class (class 1).
- **Validation approach:** Models were trained on the training set (or resampled training set) and evaluated on a separate validation set to monitor for overfitting and generalization ability. A final evaluation was performed on the unseen test set.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow/Keras
- imbalanced-learn
- Jupyter Notebook (Google Colab environment)

---

## Results

### Model Performance
**Best Model: Model 6 (Neural Network with Random Undersampling, Adam Optimizer, and Dropout) on Validation Set**

| Metric    | Score |
|-----------|-------|
| Recall    | 0.69  |
| Precision | 0.52  |
| F1 Score  | 0.59  |
| Accuracy  | 0.81  |

### Key Findings
- **Model 6** emerged as the most promising model, demonstrating a good balance between identifying actual churners (high recall) and minimizing false positives (reasonable precision) on the validation set.
- Handling class imbalance through **Random Undersampling** (Model 6) proved more effective than SMOTE (Oversampling) in achieving a better balance of precision and recall for the churn class.
- **Adam Optimizer** generally led to better model convergence and performance compared to SGD.
- **Dropout layers** helped in mitigating overfitting, contributing to better generalization on unseen data.

### Overall Observations from EDA and Modeling:
- Inactive members are significantly more likely to churn.
- Customers from Germany show a higher propensity to churn.
- Older customers and those with higher bank balances tend to churn more.

---

## Visualizations

Important charts/screenshots produced during the analysis include:
- Histograms and Box plots for numerical feature distributions.
- Bar plots for categorical feature distributions.
- Correlation heatmap of numerical features.
- Stacked bar plots showing churn rates across categorical features.
- Box plots showing churn rates across numerical features.
- Training and Validation Loss curves for each model.
- Training and Validation Recall curves for each model.
- Confusion matrices for training and validation sets of each model.

---

## Repository Structure

```
project-name/
│
├── data/                 # Raw and processed datasets
│   └── bank.csv
├── notebooks/            # Jupyter notebooks for analysis and model development
│   └── Churn_Prediction_Neural_Network.ipynb
├── src/                  # Python scripts for helper functions (if any)
├── models/               # Saved model weights or architectures
├── reports/              # Summary reports or presentations
├── images/               # Visualizations generated during EDA and model evaluation
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and documentation
```

---

## Installation

To set up the project locally, follow these steps:

```bash
git clone <repository-url> # Replace with your repository URL
cd project-name
pip install -r requirements.txt
```

The `requirements.txt` file ensures all necessary Python libraries (TensorFlow, scikit-learn, pandas, numpy, seaborn, matplotlib, imbalanced-learn) are installed with the specified versions.

## Usage

To reproduce the analysis and model development:

```bash
jupyter notebook
```

Then, open the `Churn_Prediction_Neural_Network.ipynb` notebook and run all cells sequentially. Ensure `bank.csv` is placed in the `data/` directory or update the path in the notebook.

## Future Improvements
- Deploy the final model as a web application or an API for real-time predictions.
- Explore more advanced feature engineering techniques.
- Test additional neural network architectures and hyperparameter tuning strategies (e.g., using Keras Tuner).
- Implement automated model retraining pipelines.
- Investigate other advanced sampling techniques for imbalanced datasets.

## Lessons Learned
- The critical importance of addressing class imbalance in churn prediction datasets to achieve meaningful recall for the minority class.
- Different optimizers (SGD vs. Adam) and regularization techniques (Dropout) have a significant impact on model convergence and generalization.
- Validation sets are crucial for monitoring overfitting and making informed model selection decisions.
- The choice of evaluation metric (Recall in this case) must align with the business objective (minimizing false negatives).

## Author

Samuel Mugisha D.C.

LinkedIn:[LinkedIn URL]https://

GitHub:[[[GitHub URL](https://github.com/samuelmugisha)]]()

