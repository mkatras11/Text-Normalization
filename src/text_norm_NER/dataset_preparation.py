import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
from pathlib import Path


class DatasetPrep:
    """
    A class to preprocess and prepare the dataset for training a NER with spacy.
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
        df[column_name] = df[column_name].apply(lambda x: self.preprocess_text(x))

        return df

    def extract_unique_names(self, df: pd.DataFrame, column_name: str) -> set:
        """
        Extracts a set of unique names from a specified column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the text column.
            column_name (str): The name of the column to extract unique names from.

        Returns:
            set: A set of unique names.
        """
        names_clean = set(
            name 
            for text in df[column_name]
            if isinstance(text, str)
            for name in re.split(r'/', text)
            if len(re.sub(r'[\W_]', '', name)) > 3 and not re.match(r'^[\W\d]+$', name)
        )
        return names_clean

    def filter_rows(self, df: pd.DataFrame, names_clean: set) -> pd.DataFrame:
        """
        Filters the DataFrame to keep only the rows where any name in names_clean is present in both columns.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            names_clean (set): A set of unique names to filter by.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        def filter_row(row):
            if not isinstance(row['raw_comp_writers_text'], str) or not isinstance(row['CLEAN_TEXT'], str):
                return False
            raw_names = set(re.split(r'/', row['raw_comp_writers_text']))
            clean_names = set(re.split(r'/', row['CLEAN_TEXT']))
            return not raw_names.isdisjoint(names_clean) and not clean_names.isdisjoint(names_clean)

        # Apply the filter function to the DataFrame
        return df[df.apply(filter_row, axis=1)].reset_index(drop=True)

    def preprocess_and_split_data(self, df: pd.DataFrame) -> None:
        """
        Preprocesses the DataFrame, filters the rows, splits the DataFrame into training, validation, and test sets, 
        and saves them to CSV, JSON, and SpaCy DocBin formats.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess, filter, split, and save.
            examples_path (str): Path to save the examples (training) dataset.
            test_path (str): Path to save the test dataset.
            full_path (str): Path to save the full cleaned dataset.
            val_path (str): Path to save the validation dataset.
        """
        # create folder path to save the data
        main_path = Path(__file__).resolve().parents[1] / "data"
        main_path = main_path / "data_ner"
        main_path.mkdir(parents=True, exist_ok=True)

        # Preprocess the 'raw_comp_writers_text' column
        df = self.preprocess_column(df, 'raw_comp_writers_text')

        # Extract unique names from the 'CLEAN_TEXT' column
        names_clean = self.extract_unique_names(df, 'CLEAN_TEXT')

        # Filter the DataFrame rows
        df = self.filter_rows(df, names_clean)

        # Split the data into training, test, and validation sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

        # Save the datasets to CSV files
        train_df.to_csv(main_path / "train.csv", index=False)
        test_df.to_csv(main_path / "test.csv", index=False)
        val_df.to_csv(main_path / "val.csv", index=False)
        df.to_csv(main_path / "cleaned.csv", index=False)

        # Extract unique names from the 'CLEAN_TEXT' column
        names_clean = self.extract_unique_names(df, 'CLEAN_TEXT')

        # Transform the datasets into SpaCy format
        spacy_train_data = self.transform_to_spacy_format(train_df, names_clean)
        spacy_test_data = self.transform_to_spacy_format(test_df, names_clean)
        spacy_val_data = self.transform_to_spacy_format(val_df, names_clean)

        # Save the SpaCy format datasets to JSON files
        self.save_spacy_format(spacy_train_data, main_path / "spacy_train.json")
        self.save_spacy_format(spacy_test_data, main_path / "spacy_test.json")
        self.save_spacy_format(spacy_val_data, main_path / "spacy_val.json")

        # Initialize a blank spaCy model
        nlp = spacy.blank("en")

        # Preprocess the data into spaCy's DocBin format
        doc_bin_train = self.data_preprocessing(spacy_train_data, nlp)
        doc_bin_test = self.data_preprocessing(spacy_test_data, nlp)
        doc_bin_eval = self.data_preprocessing(spacy_val_data, nlp)

        # Save the DocBin objects to disk
        doc_bin_train.to_disk(main_path / "train_data.spacy")
        doc_bin_test.to_disk(main_path / "test_data.spacy")
        doc_bin_eval.to_disk(main_path / "val_data.spacy")

    def transform_to_spacy_format(self, df: pd.DataFrame, names_clean: set) -> dict:
        """
        Transforms the DataFrame into the SpaCy training format.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            names_clean (set): A set of unique names to identify entities.

        Returns:
            dict: The transformed data in SpaCy training format.
        """
        annotations = []

        for _, row in df.iterrows():
            text = row['raw_comp_writers_text']
            entities = []
            for name in names_clean:
                start = text.find(name)
                while start != -1:
                    end = start + len(name)
                    entities.append((start, end, "PERSON"))
                    start = text.find(name, end)
            annotations.append({"text": text, "entities": entities})

        return {"annotations": annotations}

    def save_spacy_format(self, data: dict, path: str) -> None:
        """
        Saves the SpaCy training format data to a JSON file.

        Args:
            data (dict): The SpaCy training format data.
            path (str): The path to save the JSON file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def data_preprocessing(self, data: dict, spacy_model: spacy.language.Language) -> DocBin:
        """
        Preprocesses data in the training format, including Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.

        Parameters:
            data (dict): Data in the SpaCy training format.
            spacy_model (spacy.language.Language): A spaCy model object.

        Returns:
            DocBin: A spaCy DocBin object containing the processed documents.
        """
        doc_bin = DocBin()  # create a DocBin object
        for training_example in tqdm(data['annotations']): 
            text = training_example['text']
            labels = training_example['entities']
            doc = spacy_model.make_doc(text) 
            ents = []
            for start, end, label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    pass # pass if the entity span is not valid
                else:
                    ents.append(span)
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents 
            doc_bin.add(doc)
        return doc_bin


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data"

    df = pd.read_csv(path / "normalization_assesment_dataset_10k.csv")

    dataset_prep = DatasetPrep()

    dataset_prep.preprocess_and_split_data(df)