# Udacity-Project-II

### 1. ETL Pipeline
"ETL Pipeline Preparation.ipynb" - This is the python notebook that was used to form the "process_data.py". It explains how the data was cleaned.

### 2. ML Pipeline
"ML Pipeline Preparation.ipynb"- This is the python notebook that was used to form the "train_classifier.py". It explains how the model was trained and how the best parameters were obtained using grid search.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
### 3. Data
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
### 4. Models
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
### 5. App
2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

