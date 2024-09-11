# Predicting Customer Churn using ANN

This project aims to predict customer churn using Artificial Neural Networks (ANN). The project includes data preprocessing, feature engineering, model building and model evaluation.

## Data

The dataset used in this project is available at [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). The dataset contains 7043 rows and 21 columns. The columns are:

- `RowNumber`: Row number
- `CustomerId`: Customer ID
- `Surname`: Customer surname
- `CreditScore`: Customer credit score
- `Geography`: Customer geography
- `Gender`: Customer gender
- `Age`: Customer age
- `Tenure`: Customer tenure
- `Balance`: Customer balance
- `NumOfProducts`: Number of products used by customer
- `HasCrCard`: Does customer have credit card
- `IsActiveMember`: Is customer active member
- `EstimatedSalary`: Customer estimated salary
- `Exited`: Customer exited (1) or not (0)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the project, you can use the `app.py` file. You can also use Streamlit to build the web app.

```bash
streamlit run app.py
```

## Conclusion

The project demonstrates how to build an ANN model to predict customer churn. The model was trained on the [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) dataset.