"""
Amazon Product Reviews Processing

This program performs the following tasks:
1. Opens a CSV file containing Amazon product reviews.
2. Adds and populates two columns with polarity values and textual
    descriptions.
3. Saves the processed data as a new CSV file.

Dependencies:
- os
- spacy
- pandas
- textblob

Functions:
1. `load_nlp_model(model_name: str) -> spacy.language.Language`:
    Loads a spaCy language model.

2. `read_df(file_name: str) -> pd.DataFrame`:
    Reads a CSV/TXT file into a Pandas DataFrame. Includes error handling for
    FileNotFoundError, EmptyDataError, and ParserError.

3. `drop_null_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame`:
    Drops DataFrame rows containing null values in a chosen column.
    Includes error handling to ensure a valid column name.

4. `filter_and_lemmatize(nlp_model: str, input_text: str) -> str`:
    Takes a sentence and returns cleaned lemmas.
    Converts a string to an NLP object using the loaded language model,
    tokenizes, lemmatizes, and filters text.

5. `polarity_description(polarity_value: float) -> str`:
    Assigns a textual score to the polarity value.

6. `lemmatize_text_and_add_polarity_column(
    df: pd.DataFrame, target_column: str, new_column: str, nlp_model: str
    ) -> pd.DataFrame`:
        Creates and populates a new column with polarity values.
        Cleans and lemmatizes text from a chosen (target) column, calculates
        polarity value to 3 decimals using TextBlob, and assigns values to a
        new column.

7. `add_polarity_description_column(
    df: pd.DataFrame, target_column: str, new_column: str
    ) -> pd.DataFrame`:
        Creates and populates a new column with textual polarity descriptions.

8. `save_new_csv(df: pd.DataFrame, new_file_name: str)`:
    Saves DataFrame to a new CSV file.
    Includes error handling for file replacement, PermissionError,
    IsADirectoryError, EmptyDataError, and ParserError.

9. `main()`: The main program that orchestrates the entire process.

Constants:
- `language_model`: The spaCy language model name.
- `raw_dataset`: The filename of the raw Amazon product reviews dataset.
- `column_for_processing`: The target column in the dataset for processing.
- `new_column_1`: The name of the first new column containing polarity values.
- `new_column_2`: The name of the second new column containing textual
    descriptions.
- `output_csv_file_name`: The filename for the processed dataset.

Execution:
Run the main program by executing the script. Ensure that the required
dependencies are installed.
"""

import os
import spacy
import pandas as pd
from textblob import TextBlob

def load_nlp_model(model_name: str) -> spacy.language.Language:
    """
    Loads a spaCy language model.

    Parameters:
    - model_name (str): The name of the spaCy language model to load.

    Returns:
    - spacy.language.Language: The chosen spaCy language model.
    """
    return spacy.load(model_name)

def read_df(file_name: str) -> pd.DataFrame:
    """
    Reads CSV/TXT file into a Pandas DataFrame.

    Includes error handling for:
    - FileNotFoundError
    - EmptyDataError
    - ParserError

    Parameters:
    - file_name (str): A string comprising the CSV/TXT filename.

    Returns:
    - df: A Pandas DataFrame
    """
    # Open the CSV file into a DataFrame.
    try:
        df = pd.read_csv(file_name, low_memory=False)
        return df

    # Error handling for file not found.
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"File '{file_name}' not found. Please ensure the file is in the "
            "current working directory."
        ) from exc

    # Error handling for empty file.
    except pd.errors.EmptyDataError as exc:
        raise pd.errors.EmptyDataError(
            f"File '{file_name}' is empty."
        ) from exc

    # Error handling for wrong file type.
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Error reading file '{file_name}'. Please check if it is a valid "
            "CSV/TXT file."
        ) from exc

def drop_null_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Drops DataFrame rows containing null values in chosen column.

    Includes error handling to ensure valid column name.

    Parameters:
    - df (pd.DataFrame): The name of the DataFrame to be processed.
    - column_name (str): The column from which the nulls are to be dropped.

    Returns:
    - df (pd.DataFrame): The processed DataFrame.
    """
    while True:
        try:
            # Check if the specified column exists in the DataFrame
            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in the DataFrame."
                )

            # Drop rows with null values in the specified column
            df.dropna(subset=[column_name], inplace=True)
            return df # Return the modified df

        except ValueError as exc:
            print(exc)
            # Prompt the user for a new column name
            column_name = input("Please enter a valid column name: ")

def filter_and_lemmatize(nlp_model: str, input_text: str) -> str:
    """
    Takes a sentence and returns cleaned lemmas.

    Converts string to an NLP object using the loaded language model.
    Tokenizes, lemmatizes and filters text to produce a new list of relevant
    words.
    Joins the list into a str for further NLP processing.

    Parameters:
    - nlp_model (str): The loaded language model.
    - input_text (str): A string comprising text to be processed.

    Returns:
    - str: A string comprised of cleaned lemmas.
    """
    # Make text an nlp object using the small model
    sentence = nlp_model(input_text)

    tokens = [
        token.lemma_ for token in sentence       # Tokenize and lemmatize
        if not token.is_stop                     # Remove stop words
        if not token.is_punct or token.is_space  # Remove punct and whitespace
    ]

    # Join to str and make lowercase
    return ' '.join(tokens).lower()

def polarity_description(polarity_value: float) -> str:
    """
    Assigns a textual score to the polarity value.

    Parameters:
    - polarity_value (float): A float between -1.000 and 1.000.

    Returns:
    - description (str): A string comprising the textual description.
    """
    # Assign description for polarity value
    if polarity_value >= 0.800:
        description = "Extremely positive"
    elif polarity_value >= 0.400:
        description = "Very positive"
    elif polarity_value >= 0.100:
        description = "Somewhat positive"
    elif polarity_value >= -0.100:
        description = "Neutral"
    elif polarity_value >= -0.400:
        description = "Somewhat negative"
    elif polarity_value >= -0.800:
        description = "Very negative"
    elif polarity_value >= -1.000:
        description = "Extremely negative"
    return description

def lemmatize_text_and_add_polarity_column(
        df: pd.DataFrame, target_column: str, new_column: str, nlp_model: str
    ) -> pd.DataFrame:
    """
    Creates and populates a new column with polarity values.

    Cleans and lemmatizes text from a chosen (target) column.
    Calculates polarity value to 3 decimals using TextBlob.
    Assigns values to a new column.

    Parameters:
    - df (pd.DataFrame): The DataFrame for processing.
    - target_column (str): The column to be processed and analysed.
    - new_column (str): The name to be assigned to the new column.
    - nlp_model (str): The chosen spaCy language model.

    Calls:
    - filter_and_lemmatize: Takes a sentence and returns cleaned lemmas.
    
    Returns:
    - new_df (pd.DataFrame): The DataFrame with a new populated column.
    """
    # Filter and lemmatize the text in the target_column
    lemmas = (
        df[target_column]
        .apply(lambda x: filter_and_lemmatize(nlp_model, x))
    )

    # Calculate polarity to 3 decimals for each row using TextBlob
    polarity_scores = (
        lemmas.apply(lambda x: round(TextBlob(x).sentiment.polarity, 3))
    )

    # Assign the calculated polarity scores to a new column
    new_df = df.assign(**{new_column: polarity_scores})

    return new_df

def add_polarity_description_column(
        df: pd.DataFrame, target_column: str, new_column: str
    ) -> pd.DataFrame:
    """
    Creates and populates a new column with textual polarity descriptions.

    Parameters:
    - df (pd.DataFrame): The DataFrame for processing.
    - target_column (str): The column to be processed and analysed.
    - new_column (str): The name to be assigned to the new column.

    Calls:
    - polarity_description: Assigns a textual score to the polarity value.
    
    Returns:
    - new_df (pd.DataFrame): The DataFrame with a new populated column.
    """
    # Calculate polarity description for each row
    description = df[target_column].apply(polarity_description)

    # Assign the descriptions to a new column
    new_df = df.assign(**{new_column: description})

    return new_df

def save_new_csv(df: pd.DataFrame, new_file_name: str):
    """
    Saves DataFrame to new CSV file.

    Includes error handling for:
    - File already existing with user choice to replace.
    - PermissionError
    - IsADirectoryError
    - EmptyDataError
    - ParserError

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - new_file_name (str): The assigned file name.

    Returns:
    - None.
    """
    while True:
        try:
            # Check if the file already exists
            if os.path.isfile(new_file_name):
                user_input = input(
                    f"The file '{new_file_name}' already exists. Do you want "
                    "to replace it? (yes/no): "
                ).lower()
                if user_input == 'yes':
                    # Save DataFrame to a new CSV file
                    df.to_csv(new_file_name, index=False)
                    print(
                        f"DataFrame saved to '{new_file_name}' successfully."
                    )
                    break
                if user_input == 'no':
                    # Prompt the user for a new file name
                    new_file_name = input("Please enter a new file name: ")
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")

            else:
                # Save DataFrame to a new CSV file if it doesn't exist
                df.to_csv(new_file_name, index=False)
                print(f"DataFrame saved to '{new_file_name}' successfully.")
                break

        except (
            PermissionError, IsADirectoryError,
            pd.errors.EmptyDataError, pd.errors.ParserError
        ) as e:
            print(f"Error saving DataFrame to '{new_file_name}': {e}")

def main():
    """
    Program opens a CSV file, adds and populates two columns with polarity
    value as float and a textual desciption, saves as a new CSV file.
    """
    # Constants
    language_model = 'en_core_web_sm'
    raw_dataset = 'amazon_product_reviews.csv'
    column_for_processing = 'reviews.text'
    new_column_1 = 'polarity.values'
    new_column_2 = 'polarity.description'
    output_csv_file_name = 'processed_amazon_product_reviews.csv'

    # Load suitable NLP model
    nlp = load_nlp_model(language_model)

    # Read dataset into Pandas DataFrame.
    reviews_df = read_df(raw_dataset)

    # Prepare data for processing
    new_df = drop_null_rows(reviews_df, column_for_processing)

    # Calculate polarity values for product reviews and store in a new column
    new_df = lemmatize_text_and_add_polarity_column(
        new_df, column_for_processing, new_column_1, nlp
    )

    # Assign textual descriptions to polarities and store in a new column
    new_df = add_polarity_description_column(
        new_df, new_column_1, new_column_2
    )

    # Save process output as new CSV file)
    save_new_csv(new_df, output_csv_file_name)

# Run main program
if __name__ == "__main__":
    main()
