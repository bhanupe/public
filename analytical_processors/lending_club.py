import pandas as pd

from feature_engng.feature_eng import drop_highly_correlated_features
# Set the display.max_columns option to None
from visualization.visualize import heat_map_missing, univariate_analysis, bivariate_analysis, \
    multivariate_analysis, box_plot, save_show, data_correlation_plot
from wrangling.insights import explain, printline
from wrangling.prepare import encode, add_zscores, check_class_imbalance

# Imports for Deep Learning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Calculate class weights manually
from sklearn.utils import class_weight

# Optional: Plot Training History
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance

data = pd.read_csv("lending_club/data/loan_data.csv")
explain(data)
heat_map_missing(data, 'ld')

data_encoded = encode(data, int, 'lending_club/data/data_encoded.csv')
explain(data_encoded)
check_class_imbalance(data, "not.fully.paid", 0.10)

heat_map_missing(data_encoded, 'lde')

univariate_analysis(data)

bivariate_analysis(data, 'not.fully.paid', 'lde')

numeric_data = data.drop(columns='purpose')
multivariate_analysis(numeric_data, 'not.fully.paid', 'lde')
# Observation                                                   | Meaning
# Some features are strongly related (e.g., fico vs int.rate)   | Multicollinearity needs careful handling
# Most feature pairs are not perfectly separable                | Need models that can handle messy data
# Distributions are not normal, some right-skewed               | Careful feature scaling / transformation needed
# Data is imbalanced (very few defaulters)                      | Need imbalance handling during training
# Relationship between features and target is non-linear        | Not simple straight line separations

z_scores, data_with_zscores = add_zscores(data)
explain(data_with_zscores)
box_plot(z_scores, 'lde')

# Best Modeling Techniques to Use
# Technique                             | Why Suitable for This Data
# Deep Learning (Neural Networks)       | Can handle non-linear complex relationships, even if messy
# Gradient Boosting (XGBoost, LightGBM) | Very powerful with tabular data, automatically handles missing/non-linearity
# Random Forest                         | Handles non-linear splits, feature interactions automatically
# Logistic Regression (Baseline)        | Simple, interpretable — but may under perform if data is complex

# Best Practical Recommendation:
# Priority	Model
# 1	Deep Learning (Keras/TensorFlow)    — because it can model complex feature interactions and patterns easily.
# 2	XGBoost / LightGBM                  — if you want fast, highly accurate model on structured data.
# 3	Random Forest                       — as a backup simpler model.
# 4	Logistic Regression                 — as a baseline to compare.

# Deep Learning (using Keras)
# Input: all scaled numerical + encoded categorical features
# Architecture
# Layer         | Units                 | Activation
# Input Layer   | = Number of features  | —
# Dense Layer 1 | 64                    | ReLU
# Dropout       | 30%                   | —
# Dense Layer 2 | 32                    | ReLU
# Dropout       | 20%                   | —
# Output Layer  | 1                     | Sigmoid (binary classification)

# Compile with:
# Loss              = 'binary_crossentropy'
# Optimizer         = 'adam'
# Metrics           = ['accuracy', AUC]
# Handle imbalance  = class_weight option

# Train with validation set split 80/20

# Reason                                            | Explanation
# Complex non-linear feature interactions           | Deep models can capture
# Class imbalance can be adjusted via class weights | Deep models allow
# Big enough dataset (9578 rows)                    | Deep learning won’t overfit too fast
# Skewed features + messy data                      | Neural nets handle better than simple linear models

# Step  | Action
# 1     | Preprocess data (one-hot, scaling)
# 2     | Train deep learning model (MLP with 2-3 layers)
# 3     | Monitor validation AUC, not just accuracy (because of imbalance)
# 4     | Tune layers/neurons if needed
# 5     | (Optional) Try XGBoost as a comparison
#
###########################################  Full Keras Model Code  ###########################################
# Step 1: Prepare Features (X) and Target (y)
X = data_encoded.drop('not.fully.paid', axis=1)  # Drop target column from features # Input features
y = data_encoded['not.fully.paid']  # Target

# Step 2: Scale Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Handle Class Imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print(f"Class Weights: {class_weights}")

# Step 4.1 - Feature Engineering - Drop highly correlated features
X_scaled_df = pd.DataFrame(X_scaled,columns=X.columns)
data_correlation_plot(X_scaled_df,"ld")
loan_data_reduced = drop_highly_correlated_features(X_scaled_df, threshold=0.85)
data_correlation_plot(loan_data_reduced,"lddhc")

# Step 5: Build Deep Learning Model
model = Sequential()

# Input Layer + First Hidden Layer
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))

# Second Hidden Layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Step 6: Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 7: Train Model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    class_weight=class_weights,
                    verbose=1)

# Step 8: Evaluate Model
y_pred_probs = model.predict(X_test).ravel()  # Probability predictions
y_pred_classes = (y_pred_probs > 0.5).astype(int)  # Threshold 0.5

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Calculate AUC Score
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"Test AUC Score: {auc_score:.4f}")

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
save_show(plt, f'keras_accuracy_{__name__}')

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
save_show(plt, f'keras_loss_{__name__}')

# Save the full model (architecture + weights + optimizer state)
model.save('loan_default_keras_model.h5')
print("Model saved successfully to loan_default_keras_model.h5!")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Later load it
scaler = joblib.load('scaler.pkl')

# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.89      0.71      0.79      1611
#            1       0.26      0.52      0.34       305
#
#     accuracy                           0.68      1916
#    macro avg       0.57      0.62      0.57      1916
# weighted avg       0.79      0.68      0.72      1916
#
# Test AUC Score: 0.6822
# Imbalance Ratio: 5.24
# [0]	validation_0-auc:0.63817

printline()
printline()
###########################################  Keras Model Use  ###########################################
# Load the model
model = load_model('loan_default_keras_model.h5')
scaler = joblib.load('scaler.pkl')
print("Model and Scalar loaded successfully!")

data_new = pd.read_csv("lending_club/data/loan_data_new.csv")
explain(data_new)
heat_map_missing(data_new, 'ldn')

data_new_encoded = encode(data_new, int, 'lending_club/data/data_new_encoded.csv')
explain(data_new_encoded)
heat_map_missing(data_new_encoded, 'ldne')

univariate_analysis(data_new)

bivariate_analysis(data_new, 'not.fully.paid', 'ldne')

numeric_data_new = data_new.drop(columns='purpose')
multivariate_analysis(numeric_data_new, 'not.fully.paid', 'ldne')
# Observation                                                   | Meaning
# Some features are strongly related (e.g., fico vs int.rate)   | Multicollinearity needs careful handling
# Most feature pairs are not perfectly separable                | Need models that can handle messy data
# Distributions are not normal, some right-skewed               | Careful feature scaling / transformation needed
# Data is imbalanced (very few defaulters)                      | Need imbalance handling during training
# Relationship between features and target is non-linear        | Not simple straight line separations

z_scores, data_new_with_zscores = add_zscores(data_new)
explain(data_new_with_zscores)

box_plot(z_scores, 'ldne')
printline()
# Step 1: Prepare Features (X) and Target (y)
X_new = data_new_encoded.drop('not.fully.paid', axis=1)  # Drop target column from features # Input features
y_new = data_new_encoded['not.fully.paid']

# Scale features
new_data_scaled = scaler.transform(X_new)

# Now you can use it to predict
predictions = model.predict(new_data_scaled)  # (X_new_scaled = new customer data)

# Convert probabilities to 0 or 1
predicted_classes = (predictions > 0.5).astype(int)

# Create a DataFrame for easy visualization
result_df = pd.DataFrame({
    'default_probability_percent': predictions.flatten() * 100,
    'predicted_class': predicted_classes.flatten()
})
print(result_df)  # See first 10 predictions

printline()
# GENERATE CLASSIFICATION REPORT AND AUC SCORE
# Print Classification Report
print("\nClassification Report data_new:")
print(classification_report(y_new, predicted_classes))

# Calculate AUC Score
auc_score = roc_auc_score(y_new, predicted_classes)
print(f"Test AUC Score data_new: {auc_score:.4f}")
printline()
printline()
###########################################  Full XGBoost Model Code  ###########################################
# Step 3: Prepare Features (X) and Target (y)
X = data_encoded.drop('not.fully.paid', axis=1)  # Input features
y = data_encoded['not.fully.paid']  # Target

# Step 4: Scale Numerical Features (Optional for XGBoost)
# Note: XGBoost **does not require scaling**, but let's keep it for comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Handle Class Imbalance (Scale Positive Weight)
# Calculate imbalance ratio
imbalance_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

# Step 7: Build XGBoost Model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=imbalance_ratio,  # Important for handling imbalance!
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# Step 8: Train Model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# Step 9: Evaluate Model
y_pred_probs = model.predict_proba(X_test)[:, 1]  # Predicted probabilities
y_pred_classes = (y_pred_probs > 0.5).astype(int)  # Convert to class labels

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Calculate AUC Score
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"Test AUC Score: {auc_score:.4f}")

# Step 10: (Optional) Feature Importance Plot

plot_importance(model, max_num_features=10)
plt.title('Top 10 Important Features')
save_show(plt, f'xgboost_tif_{__name__}')
printline()
printline()

# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.87      0.74      0.80      1611
#            1       0.24      0.42      0.30       305
#
#     accuracy                           0.69      1916
#    macro avg       0.55      0.58      0.55      1916
# weighted avg       0.77      0.69      0.72      1916
#
# Test AUC Score: 0.6253