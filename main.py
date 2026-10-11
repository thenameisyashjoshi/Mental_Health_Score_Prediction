import joblib 
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('Mental_Health_Model.pkl')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

class StudentData(BaseModel):
    Age: int = Field(..., alias="age", ge=10, le=100)
    Gender: Literal["Male", "Female"] = Field(..., alias="gender")
    Country: str = Field(..., alias="country")
    Academic_Level: Literal["Undergraduate", "Graduate", "High School"] = Field(
        ..., alias="academic_level"
    )
    Most_Used_Platform: Literal[
        "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "YouTube",
        "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat",
    ] = Field(..., alias="most_used_platform")
    Purpose_Of_Use: Literal[
        "Networking", "Education", "Entertainment", "News",
    ] = Field(..., alias="purpose_of_use")
    Avg_Daily_Usage_Hours: float = Field(
        ..., alias="avg_daily_usage_hours", ge=0, le=24
    )
    Daily_Unlocks: int = Field(..., alias="daily_unlocks", ge=0)
    Study_Hours: float = Field(..., alias="study_hours", ge=0, le=24)
    Physical_Activity_Hours: float = Field(
        ..., alias="physical_activity_hours", ge=0, le=24
    )
    Sleep_Hours_Per_Night: float = Field(
        ..., alias="sleep_hours_per_night", ge=0, le=24
    )
    Stress_Level: Literal["Medium", "Low", "Very High", "High"] = Field(
        ..., alias="stress_level"
    )

class PredictionResponce(BaseModel):
    predicted_health_score : float

@app.get('/')
def greed():
    return{"Hello World"}


@app.post('/predict', response_model=PredictionResponce)
def predict(data : StudentData):
    country_grouped = data.Country if data.Country in top_countries else "Other"
    input = pd.DataFrame([{
    "Age"                      :data.Age,
    "Gender"                   :data.Gender,
    "Country"                  :data.Country,
    "Academic_Level"           :data.Academic_Level,
    "Most_Used_Platform"       :data.Most_Used_Platform,
    "Purpose_Of_Use"           :data.Purpose_Of_Use,
    "Avg_Daily_Usage_Hours"    :data.Avg_Daily_Usage_Hours,
    "Daily_Unlocks"            :data.Daily_Unlocks,
    "Study_Hours"              :data.Study_Hours,
    "Physical_Activity_Hours"  :data.Physical_Activity_Hours,
    "Sleep_Hours_Per_Night"    :data.Sleep_Hours_Per_Night,
    "Stress_Level"             :data.Stress_Level,
    "Group_Country"            :country_grouped

    }])

    prediction = model.predict(input)[0]
    return PredictionResponce(predicted_health_score=round(float(prediction), 2))