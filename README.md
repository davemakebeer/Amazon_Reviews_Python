# Amazon Product Reviews Processing Project

## Overview

This project is designed to process Amazon product reviews, calculate polarity values, assign textual descriptions, and ultimately enable data analysis for trends and patterns. The script provided performs data preprocessing and transformation, creating a new CSV file with enriched information. The subsequent steps involve utilizing a Jupyter Notebook for exploratory data analysis (EDA).

## Project Structure

The project consists of the following components:

1. **Amazon Product Reviews Processing Script (`amazon_reviews_processing.py`):**
   - A Python script that performs the following tasks:
     - Loads a spaCy language model.
     - Reads a CSV file containing Amazon product reviews into a Pandas DataFrame.
     - Processes the data by adding and populating two columns with polarity values and textual descriptions.
     - Saves the processed data as a new CSV file.

2. **Jupyter Notebook for Data Analysis (`amazon_reviews_analysis.ipynb`):**
   - A Jupyter Notebook dedicated to data analysis using the processed dataset.
   - Utilizes Pandas, Matplotlib, and Seaborn for exploratory data analysis.
   - Provides visualizations, statistical insights, and documentation of trends and patterns in the Amazon product reviews data.

## Dependencies

Ensure that the following Python libraries are installed before running the script or the Jupyter Notebook:

- `os`
- `spacy`
- `pandas`
- `textblob`
- `matplotlib`
- `seaborn`

You can install these dependencies using the following command:

> pip install spacy pandas textblob matplotlib seaborn

Additionally, download the spaCy language model using:

> python -m spacy download en_core_web_sm

## Project Execution

1. **Data Processing Script:**
   - Execute the `amazon_reviews_processing.py` script. This can be done by running the script in a Python environment. Ensure that the dependencies are installed before running the script.

2. **Jupyter Notebook:**
   - Open the `amazon_reviews_analysis.ipynb` notebook in Jupyter Notebook.
   - Execute each cell sequentially to perform exploratory data analysis on the processed dataset.
   - Modify the notebook to suit your specific analysis needs and objectives.

## Important Constants

Adjust the following constants in the script (`amazon_reviews_processing.py`) based on your requirements:

- `language_model`: The spaCy language model name.
- `raw_dataset`: The filename of the raw Amazon product reviews dataset.
- `column_for_processing`: The target column in the dataset for processing.
- `new_column_1`: The name of the first new column containing polarity values.
- `new_column_2`: The name of the second new column containing textual descriptions.
- `output_csv_file_name`: The filename for the processed dataset.

## Contributions and Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the project's GitHub repository. Contributions are welcome through pull requests.

Thank you for using the Amazon Product Reviews Processing Project!
