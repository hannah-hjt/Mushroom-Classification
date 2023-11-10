import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
# Read data from CSV
df = pd.read_csv("CI-EX1.csv")

# Select the features and target variable
x = df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
        'veil-type', 'veil-color', 'ring-number', 'stalk-color-below-ring', 'ring-type', 'spore-print-color', 'population', 'habitat']]
y = df['class']

# Create dummy variables for categorical columns
x = pd.get_dummies(x)

# Split the data into train, test, and validation sets
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5)

# Create a Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier on the training set
nb_classifier.fit(x_train, y_train)


# ........................................................


# Predict on the test set
y_pred_test = nb_classifier.predict(x_test)

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)

# Predict on the validation set
y_pred_val = nb_classifier.predict(x_val)

# Calculate accuracy on the validation set
accuracy_val = accuracy_score(y_val, y_pred_val)

# Calculate precision, recall, and specificity for the test set
precision_test = precision_score(y_test, y_pred_test, average=None)
precision_test_p = precision_test[1]
recall_test = recall_score(y_test, y_pred_test, average=None)
recall_test_p = recall_test[1]
confusion_matrix_test = confusion_matrix(
    y_test, y_pred_test, labels=['e', 'p'])
tn_test, fp_test, fn_test, tp_test = confusion_matrix_test.ravel()
specificity_test = tn_test / (tn_test + fp_test)

# Calculate precision, recall, and specificity for the validation set
precision_val = precision_score(y_val, y_pred_val, average=None)
precision_val_p = precision_val[1]
recall_val = recall_score(y_val, y_pred_val, average=None)
recall_val_p = recall_val[1]
confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=['e', 'p'])
tn_val, fp_val, fn_val, tp_val = confusion_matrix_val.ravel()
specificity_val = tn_val / (tn_val + fp_val)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=['e', 'p'])


# Print the scores
print("Test Set:")
print("Accuracity:", accuracy_test)
print("Precision:", precision_test_p)
print("Recall:", recall_test_p)
print("Specificity:", specificity_test)

print("\nValidation Set:")
print("Accuracity:", accuracy_val)
print("Precision:", precision_val_p)
print("Recall:", recall_val_p)
print("Specificity:", specificity_val)
print('YOUR WATCHING THE VALIDATION GRAPH')

cm_display.plot()
plt.show()
