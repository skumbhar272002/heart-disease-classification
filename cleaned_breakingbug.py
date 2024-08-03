
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as Gaussian
from sklearn.svm import SVC as SVC_Classifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import GradientBoostingClassifier as GradientBoost
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load dataset (placeholder, replace with actual dataset loading code)
df = pd.read_csv('path_to_your_dataset.csv')

# Data analysis and visualization
print('Dataset shape:', df.shape)
print('Dataset columns:', df.columns)

# Age column analysis
print('Age column summary:')
print(df['age'].describe())

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]

# Plot the histogram with custom colors
sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)
plt.show()

# Plot mean, median, and mode of age column
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color='Green')
plt.axvline(df['age'].mode()[0], color='Blue')
plt.show()

# Print the values of mean, median, and mode of age column
print('Mean:', df['age'].mean())
print('Median:', df['age'].median())
print('Mode:', df['age'].mode())

# Plot the histogram of age column using plotly and coloring by sex
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

# Find the values of sex column
print('Sex column value counts:')
print(df['sex'].value_counts())

# Calculate the percentage of male and female in the data
male_count = df[df['sex'] == 'male'].shape[0]
female_count = df[df['sex'] == 'female'].shape[0]
total_count = male_count + female_count

male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data: {female_percentage:.2f}%')

# Difference in percentages
difference_percentage = ((male_count - female_count) / female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than females in the data.')

# Find the value counts of age column grouped by sex
print('Age column value counts grouped by sex:')
print(df.groupby('sex')['age'].value_counts())

# Plot the count plot of dataset column (placeholder column, replace with actual)
fig = px.bar(df, x='dataset', color='sex')
fig.show()

# Print the values of dataset column grouped by sex (placeholder column, replace with actual)
print('Dataset column values grouped by sex:')
print(df.groupby('sex')['dataset'].value_counts())

# Hyperparameter tuning function
def hyperparameter_tuning(X, y, categorical_cols, models):
    results = {}
    for model_name, model in models.items():
        print(f'Tuning hyperparameters for {model_name}')
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'GradientBoosting':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)

        # Get best hyperparameters and evaluate on test set
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # Store results in dictionary
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNN(),
    "NB": Gaussian(),
    "SVM": SVC_Classifier(),
    "Decision Tree": DecisionTree(),
    "Random Forest": RandomForest(),
    "XGBoost": xgb.XGBClassifier(),
    "GradientBoosting": GradientBoost(),
    "AdaBoost": AdaBoost()
}

# Example usage (replace with actual data and split)
# X, y = df.drop(columns=['target']), df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# results = hyperparameter_tuning(X_train, y_train, models)
# for model_name, result in results.items():
#     print("Model:", model_name)
#     print("Best hyperparameters:", result['best_params'])
#     print("Accuracy:", result['accuracy'])
#     print()
