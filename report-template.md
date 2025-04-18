# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis is to predict the risk level associated with loan applications, specifically identifying high-risk loans (which may default) and low-risk loans (which are more likely to be repaid). By leveraging historical loan data, this analysis helps in making informed decisions on approving or rejecting loans based on the predicted risk category.


* Explain what financial information the data was on, and what you needed to predict.
The dataset provided financial information about various loan applicants. The goal was to predict the loan’s risk level (labeled as High Risk or Low Risk) based on several financial and demographic variables. The target variable, loan_status, indicates whether the loan is high-risk (1) or low-risk (0).



* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
--Target Variable (loan_status): The label we were trying to predict.

0 (Low Risk): Healthy loans likely to be repaid.

1 (High Risk): Risky loans that are likely to default.

The dataset had an imbalanced distribution, with a larger number of high-risk loans compared to low-risk loans, as seen in the following breakdown of the labels:

--High Risk: 18,765 instances

--Low Risk: 619 instances

This indicates that the dataset contains significantly more high-risk loans, which could affect the model’s performance, especially for predicting low-risk loans.



* Describe the stages of the machine learning process you went through as part of this analysis.
--Data Loading and Exploration:
We began by loading the dataset (lending_data.csv) into a pandas DataFrame and explored the first few rows to understand the data’s structure.

--Data Preprocessing:

We separated the target variable (loan_status) from the features (other financial variables).

We checked for missing values and handled any required data cleaning.

--Splitting the Data:
Using the train_test_split function from sklearn, we divided the data into training and testing sets (with a test size of 25%) to ensure the model could generalize well to unseen data.

--Model Selection and Training:
We chose Logistic Regression as the model, which is suitable for binary classification tasks. The model was trained using the training data (X_train and y_train).

--Model Evaluation:
After training, we evaluated the model’s performance on both the training and testing data using:

Accuracy Score: To get a general sense of model performance.

Confusion Matrix: To visualize how well the model predicts both low and high-risk loans.

Classification Report: To assess precision, recall, and F1-score for each class.

--Interpretation:
We analyzed the model’s predictions and discussed its strengths (strong performance on high-risk loans) and weaknesses (lower precision for low-risk loans).


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
--Logistic Regression:
Logistic regression was chosen because it is well-suited for binary classification tasks, where we have two possible outcomes (high-risk vs. low-risk loans). The model is interpretable, making it easier to understand how the features contribute to the predictions.

--Evaluation Metrics:
We used the following to assess model performance:

Accuracy: A simple metric showing the overall correctness.

Confusion Matrix: To visualize false positives and false negatives for each class.

Classification Report: To evaluate precision, recall, and F1-score for both high and low-risk classes.



## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
    --Accuracy:99%
    --Precision:
      --High Risk:0.99
      --Low Risk:0.84
    --Recall
      --High Risk:0.99
      --Low Risk:0.94



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
Based on the accuracy and evaluation metrics, Logistic Regression performs best in this case. The model achieves an overall accuracy of 99%, with perfect precision (1.00) and very high recall (0.99) for the High Risk (high-risk loans) category. However, the Low Risk (low-risk loans) category has a slightly lower precision of 0.84, indicating that the model sometimes misclassifies low-risk loans as high-risk.


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Yes, the importance of performance metrics depends on the problem at hand. In this case:

For high-risk loans, correctly identifying them (high precision and recall) is critical, as it helps prevent defaults and minimize financial losses. The logistic regression model excels here.

For low-risk loans, it may be more important to predict them accurately, especially to approve loans that are likely to be repaid. However, the model's lower precision for low-risk loans means there is a chance of approving loans that are actually high-risk.

If the cost of false positives (misclassifying low-risk loans as high-risk) is high, further adjustments or fine-tuning may be necessary.




If you do not recommend any of the models, please justify your reasoning.
Logistic Regression is the preferred model given its high accuracy, precision, and recall for high-risk loans.

If the business case prioritizes the identification of low-risk loans, or if minimizing false positives for low-risk loans is critical, you might consider improving the model by applying techniques such as class balancing (e.g., using oversampling or undersampling) or experimenting with other models (like Random Forest or XGBoost) that might handle the imbalance better.


