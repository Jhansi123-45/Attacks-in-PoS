import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import Hyperband
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score


# Load your CSV file with data
data_path = 'D:\Credit card\creditcard1.csv'  # Replace 'your_data.csv' with your file path
data = pd.read_csv(data_path)

# Assuming your parameters file has columns named 'units' and 'dropout'
X = data.drop('fraud', axis=1)  # Features
y = data['fraud']  # Target variable
cat_indices = [0,2,5,10]
oversample = BorderlineSMOTE()
X_resampled, y_resampled = oversample.fit_resample(X, y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define a function to build the model using the parameters from the Excel file
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))  # Dense layer with tunable units
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))  # Dropout layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the Hyperband tuner
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

# Get the best hyperparameters for all trials
best_trials = tuner.oracle.get_best_trials(num_trials=10)

# Create a list to store tuning parameters
tuning_parameters = []

# Store tuning parameters for each trial in the list
for i, trial in enumerate(best_trials):
    best_hps = trial.hyperparameters
    
    # Build model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, verbose=0)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_classes)
    auc_score = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_classes)
    pr_auc = auc(recall, precision)
    
    # Store all metrics in the list
    tuning_parameters.append({
        'Trial': i + 1,
        'Units': best_hps.get('units'),
        'Dropout': best_hps.get('dropout'),
        'Accuracy': accuracy,
        'AUC': auc_score,
        'PR_AUC': pr_auc
    })

# Convert the list of dictionaries to a DataFrame
parameters_df = pd.DataFrame(tuning_parameters)

# Save all tuning parameters to a single Excel file
parameters_df.to_excel('D:\\Credit card\\all_tuning_parameters1.xlsx', index=False)

# Convert the list of dictionaries to a DataFrame
parameters_df = pd.DataFrame(tuning_parameters)

# Save all tuning parameters to a single Excel file


# Now, train the final model with the best hyperparameters and get history
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate and plot confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = [0, 1]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i][j]), ha='center', va='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='AUC = %0.2f' % auc_score)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Generate Precision-Recall curve
plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
# Modify specific columns (example: modifying 'amount' column)
data['distance_from_home']= data['distance_from_home']+10  # Modify the 'amount' column values (example: doubling the values)

# Save the modified data to a new CSV file
modified_data_path = 'D:\Credit card\modified_creditcard.csv'
data.to_csv(modified_data_path, index=False)
# Load the modified CSV file
modified_data = pd.read_csv('D:\Credit card\modified_creditcard.csv')

# Assuming 'X' contains the features for prediction
X_modified = modified_data.drop('fraud', axis=1)
y_modified = modified_data['fraud']
# Use the trained model to predict fraud
# Note: Ensure 'model' is the trained fraud detection model
fraud_predictions = model.predict(X_modified)

# Print or further process the fraud predictions
print(fraud_predictions)
# Assuming 'fraud_predictions' contains the continuous output from model.predict

# Assuming 'fraud_predictions' contains the continuous output from model.predict

# Convert continuous predictions to binary class predictions
threshold = 0.5  # Adjust the threshold as needed
binary_predictions = np.where(fraud_predictions >= threshold, 1, 0)

# Ensure 'y_test' and 'binary_predictions' have the same number of samples
num_samples = min(len(y_test), len(binary_predictions))
y_test = y_test[:num_samples]
binary_predictions = binary_predictions[:num_samples]

# Now, calculate classification metrics
# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(conf_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming you already have 'conf_matrix' calculated using confusion_matrix()

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)  # Adjust font size
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, binary_predictions)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, binary_predictions)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# Plot Precision-Recall Curve
plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

