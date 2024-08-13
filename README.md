# Optimizing Heart Disease Classification: An Exploration of Multiple Algorithms and Hyperparameter Tuning

Hyperparameters and Accuracy

| Model                  | Best Hyperparameters                              | Accuracy       |
|------------------------|---------------------------------------------------|----------------|
| Logistic Regression    | {'C': 10}                                         | 0.4891         |
| KNN                    | {'n_neighbors': 7}                                | 0.6141         |
| Gaussian Naive Bayes   | {'var_smoothing': 1e-05}                          | 0.5380         |
| SVM                    | {'C': 10, 'kernel': 'rbf'}                        | 0.5978         |
| Decision Tree          | {'max_depth': 30, 'min_samples_split': 10}        | 0.6250         |
| Random Forest          | {'max_depth': 30, 'n_estimators': 200}            | 0.6087         |
| AdaBoost               | {'learning_rate': 0.1, 'n_estimators': 100}       | 0.5870         |
| Gradient Boosting      | {'learning_rate': 0.01, 'n_estimators': 200}      | 0.6250         |
| XGBoost                | {'learning_rate': 0.1, 'n_estimators': 100}       | 0.6467         |

Model Performance

| Model                    | Cross-Validation Accuracy | Test Accuracy |
|--------------------------|---------------------------|---------------|
| Logistic Regression      | 0.5115                    | 0.5109        |
| Gradient Boosting        | 0.6396                    | 0.5978        |
| KNeighbors Classifier    | 0.5767                    | 0.5870        |
| Decision Tree Classifier | 0.5840                    | 0.5761        |
| AdaBoost Classifier      | 0.6058                    | 0.5978        |
| Random Forest            | 0.6288                    | 0.6739        |
| XGBoost Classifier       | 0.6263                    | 0.6413        |
| Support Vector Machine   | 0.5877                    | 0.5870        |
| Naive Bayes Classifier   | 0.5780                    | 0.5435        |

Best Model: XGBoost Classifier
Best Model Cross-Validation Accuracy: 0.6263
Best Model Test Accuracy: 0.6413
