 
import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    s = df['categories'][0]
    start = ';'
    end = '-'
    
    # split on ';' first and create a list of categories attached to "-"
    list = s.split(start)
    
    # create empty list to be appended with cateogry column names 
    category_colnames = []
    
    for item in list:
        category_colnames.append(item.split(end)[0])
    
    for column in category_colnames:
        # set each value to be the last character of the string
        df[column] = df.categories.apply(lambda x: x.split(column+end,1)[1][0])
        # convert column from string to numeric
        df[column] = df[column].apply(pd.to_numeric, errors='ignore')
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # removing column with NaNs as all messages have been translated
    df = df.drop(['original'], axis=1)
    # removing dupes
    df = df.drop_duplicates(['id', 'message', 'genre'], keep='first')
    return df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('Msg_Category', engine, if_exists = 'replace', index=False)

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