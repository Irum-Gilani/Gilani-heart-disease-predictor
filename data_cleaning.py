
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('Heart_Disease_Dataset.csv')

# Define the columns for processing
categorical_cols = ['sex', 'cp_chestpain', 'fastingbloodsugar', 'restingrelectro', 'exerciseangia', 'slope', 'noofmajorvessels']
numerical_cols = ['age', 'restingBP', 'serumcholestrol', 'thalach_maxheartrate', 'oldpeak']

# Define the transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fit and transform the data
processed_data = preprocessor.fit_transform(data)

# Get new feature names
encoded_categories = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
transformed_columns = numerical_cols + list(encoded_categories)

# Convert the processed data to a DataFrame
processed_df = pd.DataFrame(processed_data, columns=transformed_columns)

# Save the cleaned data to a CSV file
processed_df.to_csv('Cleaned_Heart_Disease_Dataset.csv', index=False)
