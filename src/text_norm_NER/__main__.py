" Main module for the text_norm_NER package. "
import argparse
import spacy
import warnings

from text_norm_NER.dataset_preparation import DatasetPrep

def main(text: str) -> None:
    """
    Main function for the text_norm_NER package.

    Args:
    - text: str: The input text to be normalized.

    Returns:
    - str: The normalized text.
    """
    # Load the dataset
    dataset_prep = DatasetPrep()
    cleaned_text = dataset_prep.preprocess_text(text)

    # Mute warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic", message=".*torch.utils._pytree._register_pytree_node.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch", message=".*torch.cuda.amp.autocast.*")
        nlp = spacy.load("en_pipeline")
        doc = nlp(cleaned_text)


    entities = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities.append(ent.text)

    if entities:
        joined_entities = "/".join(entities)
        print(f"Normalized Text: {joined_entities}")
    else:
        print("Normalized Text:", "")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text normalization on input text.")
    parser.add_argument("text", type=str, help="The input text to be normalized.")
    args = parser.parse_args()

    main(args.text)
