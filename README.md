# Disaster Response Pipeline Project

## Project Motivation
In this project, I apply my skills in Data Engineering to analyze disaster data from Figure Eight to build a model that classifies disaster messages.


## Summary
In this project, I analyze disaster data from Figure Eight to build a model that classifies disaster messages. Data set provided by Figure Eight contains real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that I can classify these messages into different category. In this project I have built a web app where I can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 



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
    |── requirements.txt                     # contains versions of all libraries used.
    |
    └── README.md
