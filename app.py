import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px

# Read the dataset
@st.cache_data
def load_data():
    return pd.read_csv("fraudTrain.csv")

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['time_of_day'] = df['trans_date_trans_time'].dt.hour
    return df

# Features and target variable
df = load_data()
df = preprocess_data(df)

# Sample a subset of data
sample_size = 200000
df_sampled = df.sample(n=sample_size, random_state=42)

X = df_sampled[['amt', 'category', 'time_of_day']]  # Features
y = df_sampled['is_fraud']  # Target variable

# Convert category feature to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['category'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers with descriptions
classifiers = {
    "Isolation Forest": {
        "model": IsolationForest(random_state=42),
        "description": "Isolation Forest is an unsupervised anomaly detection algorithm. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature."
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "description": "Decision Tree is a supervised learning algorithm used for classification and regression tasks. It partitions the data into subsets based on the values of features and makes decisions at the internal nodes."
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "description": "Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. It combines multiple decision trees and outputs the class that is the mode of the classes output by individual trees."
    },
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42),
        "description": "Logistic Regression is a linear classification algorithm used for binary and multiclass classification tasks. It estimates probabilities using a logistic function and predicts the class with the highest probability."
    }
}

# Train and evaluate each classifier
results = {}
for name, clf_info in classifiers.items():
    clf = clf_info["model"]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        "accuracy": accuracy,
        "description": clf_info["description"],
    }

# Streamlit App
st.title("Credit Card Fraud Detection")

# Model selection dropdown
selected_model = st.selectbox("Select Model", list(classifiers.keys()))

# Model comparison
st.subheader("Model Comparison")

# Create a DataFrame to store model metrics
metrics_df = pd.DataFrame(results).T
metrics_df.drop(columns=["description"], inplace=True)

# Calculate additional classification metrics
for model_name, row in metrics_df.iterrows():
    y_pred = classifiers[model_name]["model"].predict(X_test)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    # Calculate weighted average of precision, recall, and f1-score
    precision_weighted = np.average(precision, weights=np.bincount(y_test, minlength=len(precision)))
    recall_weighted = np.average(recall, weights=np.bincount(y_test, minlength=len(recall)))
    f1_weighted = np.average(f1, weights=np.bincount(y_test, minlength=len(f1)))
    
    metrics_df.loc[model_name, "Precision"] = precision_weighted
    metrics_df.loc[model_name, "Recall"] = recall_weighted
    metrics_df.loc[model_name, "F1-score"] = f1_weighted


# Get the selected model from the dictionary
model_info = classifiers[selected_model]
model = model_info["model"]

# # Train accuracy score (assuming models are already trained)
# accuracy_score = accuracy_score(y_test, model.predict(X_test))
# st.write(f"Model Accuracy ({selected_model}):", accuracy_score)

# User input for transaction features
transaction_amount = st.number_input("Transaction Amount")
time_of_day = st.slider("Time of Day (Hour)", 0, 23, step=1)
cat_option = st.selectbox('Category of Merchant', df['category'].unique())

# Create input features for prediction
input_features = {'amt': transaction_amount, 'time_of_day': time_of_day}
for category in df['category'].unique():
    input_features['category_' + category] = 1 if category == cat_option else 0

# Make prediction using the selected model
prediction = model.predict([list(input_features.values())])[0]

# Display prediction
if prediction == 1:
    st.write(f"The transaction is predicted to be fraudulent ({selected_model}).")
else:
    st.write(f"The transaction is predicted to be legitimate ({selected_model}).")

# Display the model comparison table
st.write(metrics_df)

# Display model description
st.subheader("Model Description")
st.write(model_info["description"])

# Plot additional visualizations based on the selected model
if selected_model == "Isolation Forest":
    # Plot scatter plot showing transaction amount against time of day
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_sampled, x='time_of_day', y='amt', hue='is_fraud', palette='coolwarm')
    plt.title("Transaction Amount vs. Time of Day (Isolation Forest)")
    plt.xlabel("Time of Day (Hour)")
    plt.ylabel("Transaction Amount")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif selected_model == "Random Forest":
    # Plot ROC curve for Random Forest
    y_proba_rf = model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc_rf)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Random Forest)')
    plt.legend(loc="lower right")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif selected_model == "Logistic Regression":
    # Plot coefficients for Logistic Regression
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
    plt.title("Feature Coefficients (Logistic Regression)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif selected_model == "Decision Tree":
    X_test = X_test[X.columns]
    # Plot feature importance for Decision Tree
    feature_names = X.columns
    imp_dt = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importance_dt = sorted(zip(feature_names, imp_dt.importances_mean), key=lambda x: x[1], reverse=True)
    st.subheader("Feature Importance (Decision Tree)")
    for feature, importance in feature_importance_dt:
        st.write(f"{feature}: {importance:.4f}")

    # Plot feature importance for Decision Tree
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Create an interactive bar plot using Plotly
    fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance (Decision Tree)", template="plotly_white")

    # Update layout for better readability
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", yaxis_categoryorder="total ascending")

    # Display the plot
    st.plotly_chart(fig)
