Gilani-heart-disease-predictor
Part 1 - Deliverables 
The virtual environment file (Gilani.venv) was saved in the project directory (Gilani.Cardio) where it was created. There were no missing values in any of the 13 columns.
The ‘OneHotEncoder’ was used for encoding the nominal variables and ‘LabelEncoder’ for ordinal variables. Numerical columns (age, restingBP, serumcholestrol, thalach_maxheartrate, and oldpeak) were standardized to bring them to a similar scale (mean of 0 and variance of 1).
"Heart_Disease_Dataset" was utilized, which is available on the Kaggle. The virtual environment file (Gilani.venv) was saved in the project directory (Gilani.Cardio) where it was created. Using the jupyter notebook, a Python script (data_cleaning.py) was created in the working project directory (Gilani.Cardio). There were no missing values in any of the 13 columns.
The processed data was saved as ‘Cleaned_Heart_Disease_Dataset' into a CSV file. The uploaded Python script (data_cleaning.py) includes following data preprocessing steps:- 
•	The ‘OneHotEncoder’ was used for encoding the nominal variables and ‘LabelEncoder’ for ordinal variables. 
•	Numerical columns (age, restingBP, serumcholestrol, thalach_maxheartrate, and oldpeak) were standardized to bring them to a similar scale (mean of 0 and variance of 1).
Part 2 - Deliverables 
A Python script (model_building.py) was created which performs the following steps and prints the evaluation metrics. Steps include:-
•	Split the cleaned data into training and testing sets.
•	Choose an appropriate machine learning model (e.g., logistic regression, decision tree, etc.). 
•	Train the model on the training data. 
•	Evaluate the model on the testing data using appropriate metrics (e.g., accuracy, precision, recall)
Part 3 - Deliverables 
● A Python script (model_io.py) was created using joblib library. This script contains functions to save and load the model.
Part 4 - Deliverables 
● Google Colab was used as a development and testing environment before the final deployment stage. Colab allowed us to upload our trained ML model file directly to our Hugging Face Space ‘IrumGilani/heart-disease-predictor.’ We chose to use Gradio’s tools and libraries to build an interactive application (google_colab_processing_for_fastapi.py).
Part 5 - Deliverables 
5.1)	A Gradioor script (app.py) that provides a simple UI for the model.
5.2)	Following are the instructions for deployment of our Gradio-based application on the Hugging Face Space, ‘IrumGilani/heart-disease-predictor’:-
•	The content in the ‘requirements.txt’ file was automatically detected by the Hugging Face and the dependencies listed (gradio==5.1.0 joblib scikit-learn numpy) in it were installed. The progress of the process and error detection were monitored in the logs section of our Hugging Face Space, ‘‘IrumGilani/heart-disease-predictor.’
•	We utilized Gradio’s tools and libraries to create an interactive user frontend/interface for our machine learning model. Our setup leverages Gradio’s API-style interface within Hugging Face Space (IrumGilani/heart-disease-predictor).
•	By deploying the Gradio app on Hugging Face Space, we made both the frontend and backend accessible online, allowing users to interact with our ML seamlessly. 
•	Gradio simplifies the entire flow by allowing us to define inputs and outputs in the same code Gradio script file (app.py). Our Gradio-based API for heart disease prediction starts with the user entering health-related data through an easy-to-use web interface on Hugging Face Space, ‘IrumGilani/heart-disease-predictor.’ The backend, powered by Gradio, processes this input and feeds it into a pre-trained machine learning model that predicts the risk of heart disease. The result is then displayed back to the user, who can adjust the input to see how changes in data affect the prediction. This makes the model's insights accessible and interactive, facilitating real-time health risk assessment.
•	The ‘app.py’ was the Gradio script on Hugging Face Space, ‘IrumGilani/heart-disease-predictor.’ When Gradio app was opened on the Hugging Face Space, input data fields appeared where input data was entered for all 19 variables. After hitting Submit button, the prediction result (“Risk of heart disease”) appeared.

