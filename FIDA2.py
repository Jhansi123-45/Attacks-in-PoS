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
data_path = 'D:\Credit card\creditcard1.csv'  # Replace 'your_data.csv' with your file path
data_original = pd.read_csv(data_path)

# Modify specific columns (example: modifying 'distance_from_home' column)
data_modified = data_original.copy()
data_modified['distance_from_home'] += 10  # Modify the 'distance_from_home' column values

# Save the modified data to a new CSV file
modified_data_path = 'D:\Credit card\modified_creditcard.csv'
data_modified.to_csv(modified_data_path, index=False)

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


    return accuracy, pr_auc, fpr, tpr

# Extract features and target variable for modified data
X_modified = data_modified.drop('fraud', axis=1)
y_modified = data_modified['fraud']

# Split modified data into training and testing sets
X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(X_modified, y_modified, test_size=0.2, random_state=42)

# Extract features and target variable for original data
X_orig = data_original.drop('fraud', axis=1)
y_orig = data_original['fraud']

# Split original data into training and testing sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

# List of oversampling techniques to compare
oversamplers = {
    'No Oversampling': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, None),
    'SMOTE': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, SMOTE(random_state=42)),
    'ADASYN': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, ADASYN(random_state=42)),
    'Random Oversampling': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, RandomOverSampler(random_state=42)),
    'Borderline SMOTE': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, BorderlineSMOTE(random_state=42)),
    # Specify categorical feature indices for SMOTENC
    'SMOTENC': (X_train_orig, X_test_orig, y_train_orig, y_test_orig, SMOTENC(categorical_features=[0, 2, 5, 10], random_state=42))
}

# Dictionary to store evaluation results
results = {}

# Train and evaluate models for each oversampling technique on both original and modified data
for method, (X_train, X_test, y_train, y_test, oversampler) in oversamplers.items():
    accuracy, pr_auc, fpr, tpr = train_evaluate_model(X_train, X_test, y_train, y_test, oversampler)
    results[method] = {'Accuracy': accuracy, 'PR AUC': pr_auc, 'FPR': fpr, 'TPR': tpr}

# Plot comparison of evaluation metrics
plt.figure(figsize=(12, 8))

# Plot accuracy comparison
plt.subplot(2, 2, 1)
accuracy_scores = [results[method]['Accuracy'] for method in results]
plt.bar(results.keys(), accuracy_scores, color='skyblue')
plt.xlabel('Oversampling Technique')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

# Plot PR AUC comparison
plt.subplot(2, 2, 2)
pr_auc_scores = [results[method]['PR AUC'] for method in results]
plt.bar(results.keys(), pr_auc_scores, color='salmon')
plt.xlabel('Oversampling Technique')
plt.ylabel('PR AUC')
plt.title('PR AUC Comparison')

# Plot ROC curve comparison
plt.subplot(2, 2, 3)
for method in results:
    fpr, tpr = results[method]['FPR'], results[method]['TPR']
    plt.plot(fpr, tpr, label=f'{method} (AUC = {auc(fpr, tpr):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()

plt.tight_layout()
plt.show()
