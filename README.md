# Disaster Response Pipeline Project
https://view6914b2f4-3001.udacity-student-workspaces.com/



## Project Motivation

In this project, I apply skills I learned in Data Engineering Section to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - (Run run.py directly if DisasterResponse.db and classifier.pkl already exist.)

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    

3. Go to http://0.0.0.0:3001/


## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   ├── train_classifier.py              # Train ML model
    │   └── classifier.pkl                   # pikkle file of model   
    |   
    └── README.md

