import sys

import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """
    load data from CSV files

    Parameters:
    messages_filepath (str): messages csv file path
    categories_filepath (str): categories csv file path 

    
    Returns:
    
    df (pandas dataframe): dataframe that contains message and categories
    data
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean data from dataframe
        - split categories into separated clearly named columns
        - convert to binary
        - drop duplicates

    Parameters:
    df (pandas dataframe): the dataframe to be clean

    Returns:
    df (pandas dataframe): a cleaned dataframe
    """
    
    categories =  df.categories.str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save dataframe in a sql database

    Parameters:
    df (pandas dataframe): the dataframe to be stored
    database_filename (str): sqlite database file name
    
    Returns:
    None
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DisasterResponse", engine, index=False, if_exists='replace')  


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