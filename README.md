# Optimizing Heart Disease Classification: An Exploration of Multiple Algorithms and Hyperparameter Tuning

This project explores various machine learning algorithms and their hyperparameter tuning to classify heart disease. The goal is to find the best-performing model based on accuracy metrics by evaluating models like Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.

## Project Overview

Heart disease classification is a critical application of machine learning in the healthcare industry. This project focuses on optimizing multiple machine learning models to classify the presence of heart disease. The dataset used includes various health metrics such as age, sex, cholesterol levels, and more. The performance of different models is evaluated based on accuracy, both on cross-validation and test data.

## Dataset

The dataset used in this project contains the following fields:

- **ID**: Unique identifier for the dataset entry.
- **Age**: Age of the patient.
- **Sex**: Gender of the patient.
- **CP**: Chest pain type.
- **Trestbps**: Resting blood pressure.
- **Chol**: Cholesterol level.
- **Fbs**: Fasting blood sugar.
- **Restecg**: Resting electrocardiographic results.
- **Thalch**: Maximum heart rate achieved.
- **Exang**: Exercise-induced angina.
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **Slope**: The slope of the peak exercise ST segment.
- **Ca**: Number of major vessels colored by fluoroscopy.
- **Thal**: Thalassemia test result.
- **Num**: Diagnosis of heart disease (target variable).

The dataset can be found [here](dataset.csv).

## Objective

The main objective of this project is to identify the best machine learning model for classifying heart disease by experimenting with various algorithms and hyperparameter tuning. The best model is chosen based on cross-validation accuracy and test accuracy.

## Methodology

1. **Data Preprocessing**: The dataset is cleaned, and features are standardized where necessary.
2. **Model Training**: Several machine learning models, including:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - AdaBoost
   - Gradient Boosting
   - XGBoost
3. **Hyperparameter Tuning**: Each model's hyperparameters are tuned using grid search to optimize performance.
4. **Model Evaluation**: Models are evaluated based on accuracy during cross-validation and on the test set.

## Features

- **Hyperparameter Tuning**: Grid search is used to fine-tune the model parameters.
- **Cross-Validation**: Used to evaluate the model's performance and reduce overfitting.
- **Model Comparison**: Multiple models are compared to find the best-performing one.

## Prerequisites

- Python 3.x
- Libraries: 
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `seaborn`

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/heart-disease-classification.git
   cd heart-disease-classification
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - The dataset can be found [here](dataset.csv).
   - Place the `dataset.csv` file in the project directory.

4. **Run the main script**:
   ```bash
   python main.py
   ```

## Results

| Model                    | Cross-Validation Accuracy | Test Accuracy |
|--------------------------|---------------------------|---------------|
| Logistic Regression      | 0.8115                    | 0.8109        |
| Gradient Boosting        | 0.9396                    | 0.8978        |
| KNeighbors Classifier    | 0.8767                    | 0.8870        |
| Decision Tree Classifier | 0.8840                    | 0.8761        |
| AdaBoost Classifier      | 0.9058                    | 0.8978        |
| Random Forest            | 0.9288                    | 0.9339        |
| XGBoost Classifier       | 0.9263                    | 0.9413        |
| Support Vector Machine   | 0.8877                    | 0.8870        |
| Naive Bayes Classifier   | 0.8780                    | 0.8435        |

- **Best Model**: XGBoost Classifier
  - **Cross-Validation Accuracy**: 0.9263
  - **Test Accuracy**: 0.9413

## Usage

The `main.py` script handles the training, hyperparameter tuning, and evaluation of all the models. You can modify the script to change the dataset or the models being used.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors of open-source machine learning libraries and the healthcare community for providing valuable insights and datasets to help with this analysis.
