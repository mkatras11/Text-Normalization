"A module to preprocess text data by removing noisy names, cleaning delimiters, and handling empty rows."
import re
import pandas as pd

class TextPreprocessor:
    """
    A class to preprocess text data by removing noisy names, cleaning delimiters, and handling empty rows.
    """

    # Define noisy names to remove from the text
    NOISY_NAMES = [
        r'<Unknown>', r'Copyright Control', r'Traditional', r'COPYRIGHT CONTROL', 
        r'\[traditional\]', r'WRITER UNKNOWN', r'Not Documented', 
        r'Sonoton Music GmbH & Co. KG', r'UNKNOWN WRITER \(999990\)', 
        r'UNKNOWN WRITER', r'PUBLISHER UNKNOWN', 
        r'Figurata Music GmbH & Co. KG', r'#unknown#'
    ]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses a single string by:
        1. Removing frequently used noisy names.
        2. Cleaning extra delimiters.
        3. Removing leading/trailing delimiters and spaces.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The cleaned and preprocessed text.
        """
        # If the input is not a string, return it as is
        if not isinstance(text, str):
            return text

        # removeleading and trailing quotes
        text = re.sub(r'^"|"$', '', text)
    
        # Remove noisy names
        for noisy in self.NOISY_NAMES:
            text = re.sub(noisy, '', text, flags=re.IGNORECASE)

        # Remove extra delimiters
        text = re.sub(r'[/,]\s*$', '', text)

        # Remove double delimiters (e.g., //)
        text = re.sub(r'//', '/', text)

        # Remove leading and trailing delimiters
        text = re.sub(r'^/|/$', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text if text else ""

    def preprocess_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Preprocesses a text column in a DataFrame by removing frequently used noisy names, cleaning delimiters,
        and removing empty rows.

        Args:
            df (pd.DataFrame): The DataFrame containing the text column.
            column_name (str): The name of the column to preprocess.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        # Preprocess the text column
        df[column_name] = df[column_name].apply(lambda x: self.preprocess_text(x))

        return df

    def split_and_save_data(self, df: pd.DataFrame, examples_path: str, test_path: str, full_path: str) -> None:
        """
        Splits the DataFrame into training and test sets, and saves them to CSV files.

        Args:
            df (pd.DataFrame): The DataFrame to split and save.
            examples_path (str): Path to save the examples (training) dataset.
            test_path (str): Path to save the test dataset.
            full_path (str): Path to save the full cleaned dataset.
        """
        # Split the data into examples and test sets
        df_examples = df.sample(frac=0.01, random_state=200)
        df_test = df.drop(df_examples.index)
        # get 1000 samples for testing the solution
        df_test = df_test.sample(n=1000, random_state=200)
        # drop rows with np.nan values in CLEAN_TEXT column
        df_test = df_test.dropna(subset=['CLEAN_TEXT'])
        # Save the cleaned DataFrames to CSV files
        df_examples.to_csv(examples_path, index=False)
        df_test.to_csv(test_path, index=False)
        df.to_csv(full_path, index=False)
