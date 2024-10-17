from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

# Load your trained ML model
model = joblib.load("random_forest_model.joblib")


# Define the input data model
class PredictionInput(BaseModel):
    age: float
    restingBP: float
    serumcholestrol: float
    thalach_maxheartrate: float
    sex_1: float
    cp_chestpain_1: float
    cp_chestpain_2: float
    cp_chestpain_3: float
    fastingbloodsugar_1: float
    restingrelectro_1: float
    restingrelectro_2: float
    exerciseangia_1: float
    slope_1: float
    slope_2: float
    slope_3: float
    noofmajorvessels_1: float
    noofmajorvessels_2: float
    noofmajorvessels_3: float


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Prepare the data for the model
    input_features = [[
        input_data.age, input_data.restingBP, input_data.serumcholestrol,
        input_data.thalach_maxheartrate, input_data.sex_1, input_data.cp_chestpain_1,
        input_data.cp_chestpain_2, input_data.cp_chestpain_3, input_data.fastingbloodsugar_1,
        input_data.restingrelectro_1, input_data.restingrelectro_2, input_data.exerciseangia_1,
        input_data.slope_1, input_data.slope_2, input_data.slope_3,
        input_data.noofmajorvessels_1, input_data.noofmajorvessels_2, input_data.noofmajorvessels_3
    ]]

    # Make prediction
    prediction = model.predict(input_features)
    return {
        "prediction": prediction[0],
        "confidence": "High" if prediction[0] > 0.8 else "Low",
        "description": "Risk of heart disease" if prediction[0] > 0.5 else "No significant risk"
    }
