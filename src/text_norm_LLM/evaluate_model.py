import os
from pathlib import Path
import dotenv
import numpy as np
import pandas as pd
from text_norm_LLM.text_norm import TextNormalizer
from text_norm_LLM.data_preproc import TextPreprocessor
from sklearn.metrics import accuracy_score

def calculate_accuracy(ground_truths, predictions):
    """
    Calculate the accuracy between two lists, treating nan values as empty strings.

    Parameters:
    ground_truths (list): List of ground truth values (strings and/or nan).
    predictions (list): List of predicted values (strings and/or nan).

    Returns:
    float: Accuracy between 0 and 1.
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("The two lists must have the same length.")

    # Replace nan values with empty strings
    ground_truths = ["" if isinstance(x, float) and np.isnan(x) else x for x in ground_truths]
    predictions = ["" if isinstance(x, float) and np.isnan(x) else x for x in predictions]

    # Calculate accuracy
    correct = sum(gt.lower() == pred.lower() for gt, pred in zip(ground_truths, predictions))

    accuracy = correct / len(ground_truths)

    return accuracy


def evaluate_model(csv_path: Path, normalizer: TextNormalizer) -> float:
    """
    Evaluate the model on a subset of the dataframe with 1000 samples.

    Args:
        csv_path (Path): Path to the CSV file containing the test samples.
        normalizer (TextNormalizer): The TextNormalizer instance to use for normalization.

    Returns:
        float: The accuracy of the model on the test set.
    """
    # Load the test data
    test_df = pd.read_csv(csv_path)

    # Ensure the required columns exist
    if 'raw_comp_writers_text' not in test_df.columns or 'CLEAN_TEXT' not in test_df.columns:
        raise ValueError("CSV must contain 'raw_comp_writers_text' and 'CLEAN_TEXT' columns.")

    # Initialize lists to store predictions and ground truth
    predictions = []
    ground_truth = []
    pred = pd.DataFrame(columns=['predicted', 'ground_truth'])

    # Iterate over the test samples
    for _, row in test_df.iterrows():
        query = row['raw_comp_writers_text']
        true_value = row['CLEAN_TEXT']

        if query and not isinstance(query, float):
            response = normalizer.normalize_text(query)
            predicted_value = response.normalized_text[0].CLEAN_TEXT if response.normalized_text else np.nan
        else:
            predicted_value = np.nan

        # Append to lists
        predictions.append(predicted_value)
        ground_truth.append(true_value)

    # Calculate accuracy
    accuracy = calculate_accuracy(ground_truth, predictions)
  
    pred['predicted'] = predictions
    pred['ground_truth'] = ground_truth

    #save predictions
    path = Path("data") / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    failed_pred = pred[pred['predicted'] != pred['ground_truth']]
    correct_pred = pred[pred['predicted'] == pred['ground_truth']]  
    failed_pred.to_csv(path / "failed_predictions.csv", index=False)
    correct_pred.to_csv(path / "correct_predictions.csv", index=False)

    return accuracy

def main():
    # Load environment variables
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Initialize the TextPreprocessor
    preprocessor = TextPreprocessor()

    # Define data directory
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

    # Evaluate the model on the test set
    accuracy = evaluate_model(test_path, normalizer)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()