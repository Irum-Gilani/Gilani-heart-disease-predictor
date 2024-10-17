# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.get_params()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('Cleaned_Heart_Disease_Dataset_with_target.csv')

# Separate features and target variable
X = data.drop('target', axis=1)  # Features
y = data['target']               # Target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training data
rfc.fit(X_train,y_train)

# Make predictions on the test data
predicted_y = rfc.predict(X_test)
predicted_y
rfc.score(X_train,y_train)
rfc.score(X_test,y_test)
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
for i in range(10, 100, 10):
    print(f"Running model with {i} estimators")
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    accuracy = rfc.score(X_test, y_test)
    print(f"Accuracy is {accuracy}")
