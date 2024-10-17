# Import necessary libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to save the trained model to a file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Function to load the trained model from a file
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# Example code that could be used for training and saving:
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('Cleaned_Heart_Disease_Dataset_with_target.csv')
    
    # Separate features and target variable
    X = data.drop('target', axis=1)  # Features
    y = data['target']               # Target
    
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Classifier
    rfc = RandomForestClassifier()
    
    # Train the model on the training data
    rfc.fit(X_train, y_train)
    
    # Save the model to a file
    save_model(rfc, 'random_forest_model.joblib')
    
    # Make predictions on the test data
    predicted_y = rfc.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predicted_y)
    precision = precision_score(y_test, predicted_y)
    recall = recall_score(y_test, predicted_y)
    f1 = f1_score(y_test, predicted_y)
    
    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Testing with different numbers of estimators
    for i in range(10, 100, 10):
        print(f"Running model with {i} estimators")
        rfc = RandomForestClassifier(n_estimators=i, random_state=42)
        rfc.fit(X_train, y_train)
        test_accuracy = rfc.score(X_test, y_test)
        print(f"Accuracy is {test_accuracy:.2f}")
