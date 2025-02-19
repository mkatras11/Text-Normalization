import os
from pathlib import Path
import dotenv
import pandas as pd
from .text_norm import TextNormalizer
from .data_preproc import TextPreprocessor


def main():
    # Load environment variables
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Initialize the TextPreprocessor
    preprocessor = TextPreprocessor()

    data_dir = Path("data")
    # Load the dataset
    df = pd.read_csv(data_dir / "normalization_assesment_dataset_10k.csv")

    # Preprocess the text column
    df = preprocessor.preprocess_column(df, 'raw_comp_writers_text')

    # Define file paths using pathlib
    examples_path = data_dir / "normalization_assesment_dataset_10k_cleaned_examples.csv"
    test_path = data_dir / "normalization_assesment_dataset_10k_cleaned_test.csv"
    full_path = data_dir / "normalization_assesment_dataset_10k_cleaned.csv"

    # Ensure the directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if any of the files exist
    if not all(path.exists() for path in [examples_path, test_path, full_path]):
        # Split and save the data only if the files do not exist
        preprocessor.split_and_save_data(df, examples_path, test_path, full_path)

    # Load examples
    examples = TextNormalizer.load_examples(examples_path)

    # Initialize the TextNormalizer with examples
    normalizer = TextNormalizer(openai_api_key=OPENAI_API_KEY, examples=examples)

    # Normalize a query
    query = "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)"
    query = TextPreprocessor().preprocess_text(query)

    if query:
        response = normalizer.normalize_text(query)
        for normalized_text in response.normalized_text:
            print(f"Normalized Text: {normalized_text.CLEAN_TEXT}")

    else:
        print("Normalized Text: nan")


if __name__ == "__main__":
    main()
