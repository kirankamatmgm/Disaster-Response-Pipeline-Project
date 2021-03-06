import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message and categories files 
    merge them and return the new dataframe
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        df: The merged dataset
    """  
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, left_on='id', right_on='id', how='outer')
    return df


def clean_data(df):
    """Clean the merged dataframe to make it ready to analyze
    Input: DataFrame
    Output: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df['categories']).str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  pd.Series(categories[column]).str.split('-').str[1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe into the given database 
    input: 
        df: dataframe
        database_filename: database to store the cleaned dataframe 
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()