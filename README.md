# Gilani-heart-disease-predictor 
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
● A Python script (api.py) was created that defines the Gradio-based FastAPI application and the prediction endpoint.
Part 5 - Deliverables 
•	A Gradio script (app.py) was created that provides a simple UI for the model. 
•	 Following are the instructions for deployment of our Gradio-based application on the Hugging Face Space, ‘IrumGilani/heart-disease-predictor’:-
•	Google Colab was used as a development and testing environment before the final deployment stage. Colab allowed us to upload our trained ML model file (model.joblib) directly to our Hugging Face Space ‘IrumGilani/heart-disease-predictor,’ for deployment. Once uploaded, Commit changes was clicked to save the file to our Space. Hugging Face Space automatically detected the file and deployed our Gradio application.
