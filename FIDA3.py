import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SMOTENC
from keras_tuner.tuners import Hyperband
from tensorflow import keras
from tensorflow.keras import layers

# Load original CSV file with data
data_path = 'D:\Credit card\creditcard1.csv'
data_original = pd.read_csv(data_path)

# Function to introduce false data injection
def inject_false_data(data):
    false_data = data[data['fraud'] == 0].sample(n=500, replace=True)
    false_data['fraud'] = 1
    return pd.concat([data, false_data], ignore_index=True)

# Inject false data into the original dataset
data_modified = inject_false_data(data_original)

# Function to train and evaluate models
def train_evaluate_model(X_train, X_test, y_train, y_test, oversampler=None):
    # Define a function to build the model
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
        model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))  # Dense layer with tunable units
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))  # Dropout layer
        model.add(layers.Dense(1, activation='sigmoid'))  # Output layer

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    if oversampler:
        # Apply oversampling
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        X_train, y_train = X_resampled, y_resampled

    # Build the Keras model
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='D:\\Credit card\\tuner_dir4',
        project_name='D:\\Credit card\\fraud_detection4'
    )
    
    # Perform hyperparameter tuning
    tuner.search(X_train, y_train, validation_split=0.2, epochs=20)

    # Get the best hyperparameters for the ANN model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=20)

    # Evaluate the tuned ANN model on the test set
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Binarize predictions using a threshold

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_binary)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    return accuracy, pr_auc, fpr, tpr, best_hps

# Function to calculate metrics for a dataset
def calculate_metrics(X, y):
    # Extract features and target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of oversampling techniques to compare (excluding 'No Oversampling')
    oversamplers = {
        'SMOTE': (X_train, X_test, y_train, y_test, SMOTE(random_state=42)),
        'ADASYN': (X_train, X_test, y_train, y_test, ADASYN(random_state=42)),
        'Random Oversampling': (X_train, X_test, y_train, y_test, RandomOverSampler(random_state=42)),
        'Borderline SMOTE': (X_train, X_test, y_train, y_test, BorderlineSMOTE(random_state=42)),
        # Specify categorical feature indices for SMOTENC
        'SMOTENC': (X_train, X_test, y_train, y_test, SMOTENC(categorical_features=[0, 2, 5, 10], random_state=42))
    }

    # Dictionary to store evaluation results
    results = {}

    # Train and evaluate models for each oversampling technique
    for method, (X_train, X_test, y_train, y_test, oversampler) in oversamplers.items():
        accuracy, pr_auc, _, _, best_hps = train_evaluate_model(X_train, X_test, y_train, y_test, oversampler)
        results[method] = {'Accuracy': accuracy, 'PR AUC': pr_auc, 'Hyperparameters': best_hps}

    return results

# Calculate metrics for the original data
original_metrics = calculate_metrics(data_original.drop('fraud', axis=1), data_original['fraud'])

# Calculate metrics for the modified data
modified_metrics = calculate_metrics(data_modified.drop('fraud', axis=1), data_modified['fraud'])

# Generate tables for performance metrics and model hyperparameters
original_metrics_df = pd.DataFrame(original_metrics).T
modified_metrics_df = pd.DataFrame(modified_metrics).T

print("Original Data Metrics:")
print(original_metrics_df)
print("\nModified Data Metrics:")
print(modified_metrics_df)
