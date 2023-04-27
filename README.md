# Udacity-Project-II - The Disaster Response Pipeline Summary

## Introduction:
The Disaster Response tool was built to classify real messages that were sent during disaster events into topic categories.

## Contents:
### 1. ETL Pipeline
"ETL Pipeline Preparation.ipynb" - This is the python notebook that was used to form the "process_data.py". It explains how the data was cleaned.

### 2. ML Pipeline
"ML Pipeline Preparation.ipynb"- This is the python notebook that was used to form the "train_classifier.py". It explains how the model was trained and how the best parameters were obtained using grid search.

### 3. Data

### 4. Models
        
### 5. App

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`
        
2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

