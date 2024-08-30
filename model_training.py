import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import streamlit as st

# Function to train the model and make predictions
def train_and_predict(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Define the full preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Define feature selection
    feature_selector = SelectKBest(score_func=f_classif, k=10)

    # Combine preprocessing and feature selection in a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', feature_selector)
    ])

    # Fit the pipeline and transform training data
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    print(f"Training data transformed shape: {X_train_transformed.shape}")

    # Transform the test data
    X_test_transformed = pipeline.transform(X_test)
    print(f"Test data transformed shape: {X_test_transformed.shape}")

    # Initialize classifiers
    ada_model = AdaBoostClassifier()
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    # Fit the classifiers
    ada_model.fit(X_train_transformed, y_train)
    rf_model.fit(X_train_transformed, y_train)
    gb_model.fit(X_train_transformed, y_train)

    # Create a voting classifier
    voting_model = VotingClassifier(estimators=[
        ('ada', ada_model),
        ('rf', rf_model),
        ('gb', gb_model)],
        voting='soft')

    # Fit the voting classifier
    voting_model.fit(X_train_transformed, y_train)

    # Predict and evaluate
    y_pred = voting_model.predict(X_test_transformed)
    train_acc = accuracy_score(y_train, voting_model.predict(X_train_transformed))
    test_acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Get feature names from the pipeline
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    mask = pipeline.named_steps['feature_selector'].get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    return train_acc, test_acc, conf_matrix, class_report, X_test, y_test, y_pred, voting_model, selected_feature_names, pipeline

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Function to get feature importances from the voting model
def get_feature_importances(voting_model, feature_names):
    importances = np.zeros(len(feature_names))
    for name, estimator in voting_model.named_estimators_.items():
        if hasattr(estimator, 'feature_importances_'):
            importances += estimator.feature_importances_
    importances /= len(voting_model.named_estimators_)
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)