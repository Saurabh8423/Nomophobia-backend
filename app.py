import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.svm import SVR
from fastapi.middleware.cors import CORSMiddleware

# Load the trained machine learning model
svr_model = SVR(kernel='rbf')

# Define scoring system and symptom scores
scoring_system = {
    'Age': {'15-17 Years': 5, '18-22 Years': 4, '23-25 Years': 3, '25 and Above': 1},
    'Gender': {'Male': 3, 'Female': 3},
    'Time': {'0-2 hours': 1, '3-4 hours': 2, '5-7 hours': 3, '8-10 hours': 4, '11-13 hours': 5, '14 and above': 6},
    'Check_Social_Media': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1},
    'Boring_Studies': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1},
    'No_Fun': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1},
    'Skip_Activities': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1},
    'Forgetful': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1},
    'Deprive_Sleep': {'Strongly Agree': 3, 'Agree': 2, 'Neutral': 1, 'Disagree': 0, 'Strongly Disagree': -1}
}

# Symptom scores
symptom_scores = {'Headache': 3, 'Eye Problem': 3, 'Fever': 3, 'Frustrated': 3, 'Anxiety': 3, 'Others': 3}

# Function to score symptoms
def score_symptoms(symptoms):
    total_score = 0
    for symptom in symptoms.split(', '):
        total_score += symptom_scores.get(symptom, 0)
    return total_score

# Define the function to predict the nomophobia score
def predict_nomophobia_score(user_input):
    nomophobia_score = (
        scoring_system['Age'][user_input['Age']] +
        scoring_system['Gender'][user_input['Gender']] +
        scoring_system['Time'][user_input['Time']] +
        score_symptoms(user_input['Symptoms']) +
        scoring_system['Check_Social_Media'][user_input['Check_Social_Media']] +
        scoring_system['Boring_Studies'][user_input['Boring_Studies']] +
        scoring_system['No_Fun'][user_input['No_Fun']] +
        scoring_system['Skip_Activities'][user_input['Skip_Activities']] +
        scoring_system['Forgetful'][user_input['Forgetful']] +
        scoring_system['Deprive_Sleep'][user_input['Deprive_Sleep']]
    )
    return nomophobia_score

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your React.js app runs on this port
    "http://localhost:3001",  # Assuming your FastAPI app runs on this port
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",
    "https://smartphone-addiction-care.vercel.app"  # Your deployed frontend
]

# Add CORS support to your FastAPI app with specific origins allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allows specified methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for user input
class UserInput(BaseModel):
    Age: str
    Gender: str
    Time: str
    Symptoms: str
    Check_Social_Media: str
    Boring_Studies: str
    No_Fun: str
    Skip_Activities: str
    Forgetful: str
    Deprive_Sleep: str

# Define root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Define prediction endpoint
@app.post("/predict")
async def predict(user_input: UserInput):
    # Convert user input to dictionary
    user_input_dict = user_input.dict()

    # Predict nomophobia score
    nomophobia_score = predict_nomophobia_score(user_input_dict)

    # Return the predicted nomophobia score as JSON response
    return JSONResponse(content={'nomophobia_score': nomophobia_score})

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Get the port from the environment variable or use 8000
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
