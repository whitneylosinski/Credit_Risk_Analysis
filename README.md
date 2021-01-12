# Credit Risk Analysis
The purpose of this analysis was to use and compare different machine learning methods to predict credit risk based on different account criteria and determine the best method.

### Results
Below are the results of each machine learning method used for predicting credit risk from the dataset.  The balanced accuracy score was calculated using the balanced_accuracy_score method from sklearn.metrics.  The precision, recall and F1 scores were calculated from the confusion matrix and the imbalanced classification report is provided to verify the results.
- **Naive Random Oversampling**
  - Balanced Accuracy Score: 67.70%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |1.09%          |76.24%      |2.15%   |
    |Low Risk   |99.76%         |59.17%      |74.28%  |
  - Imbalanced Classification Report: ![Oversampling](Results/Oversampling.png)
- **Smote Oversampling**
  - Balanced Accuracy Score: 66.24%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |1.20%          |63.37%      |2.35%   |
    |Low Risk   |99.69%         |69.11%      |81.63%  |
  - Imbalanced Classification Report: ![Smote](Results/Smote.png)
- **Cluster Centroids Undersampling**
  - Balanced Accuracy Score: 54.70%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |0.68%          |68.32%      |1.35%   |
    |Low Risk   |99.55%         |41.08%      |58.16%  |
  - Imbalanced Classification Report: ![Undersampling](Results/Undersampling.png)
- **SMOTEEN Over and Under Sampling**
  - Balanced Accuracy Score: 64.47%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |0.98%          |72.28%      |1.92%   |
    |Low Risk   |99.71%         |56.67%      |72.27%  |
  - Imbalanced Classification Report: ![SMOTEEN](Results/SMOTEEN.png)
- **Balanced Random Forest Classifier**
  - Balanced Accuracy Score: 78.85%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |3.19%          |70.30%      |6.11%   |
    |Low Risk   |99.80%         |87.41%      |93.20%  |
  - Imbalanced Classification Report: ![RandomForest](Results/RandomForest.png)
- **Easy Ensemble Adaboost Classifier**
  - Balanced Accuracy Score: 93.17%
  - Calculated Precision, Recall and F1 Scores
    |Loan Status|Precision Score|Recall Score|F1 Score|
    |:---:      |:---:          |:---:       |:---:   |
    |High Risk  |8.64%          |92.08%      |15.80%  |
    |Low Risk   |99.95%         |94.25%      |97.02%  |
  - Imbalanced Classification Report: ![EasyEnsemble](Results/EasyEnsemble.png)

### Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
- The loan was predicted to be high risk.  What are the chances the loan is high risk?  Very low
- The loan was predicted to be low risk.  What are the chances the loan in low risk? Very high
- The loan is actually high risk.  What are the changes the loan is classified as high risk? Fairly high
- The loan is actually low risk.  What are the chances the loan is classified as low risk? Fairly high.
