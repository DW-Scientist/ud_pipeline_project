import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# reading the files within the data directory
def load_data(messages_filepath, categories_filepath):
    message_data = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(message_data, categories, on="id")
    return df


# prepare the data for our alghorithm
def clean_data(df):
    cats = df["categories"].str.split(";", expand=True)
    r = cats.iloc[[1]]
    cat_cols = [c.split("-")[0] for c in r.values[0]]
    cats.columns = cat_cols

    for col in cats:
        cats[col] = cats[col].str[-1]
        cats[col] = cats[col].astype(np.int)

    df = df.drop("categories", axis=1)
    df = pd.concat([df, cats], axis=1)
    df = df.drop_duplicates()

    return df


# save the data to our database for the our algorithm
def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    table_name = database_filename.replace(".db", "") + "_table"
    table_name = table_name.replace("data/", "")
    df.to_sql(table_name, engine, index=False, if_exists="replace")


# writing a main function to execute everything together
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    print("start data preprocessing...")
    main()
