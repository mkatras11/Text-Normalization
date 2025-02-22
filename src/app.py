from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import warnings
import spacy
import dotenv
import pandas as pd
from pathlib import Path

from text_norm_LLM.text_norm import TextNormalizer
from text_norm_LLM.data_preproc import TextPreprocessor
from text_norm_NER.dataset_preparation import DatasetPrep

app = FastAPI()

# Load environment variables
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the TextPreprocessor
preprocessor = TextPreprocessor()

# Define file paths using pathlib
data_dir = Path("data")
examples_path = data_dir / "normalization_assesment_dataset_10k_cleaned_examples.csv"
test_path = data_dir / "normalization_assesment_dataset_10k_cleaned_test.csv"
full_path = data_dir / "normalization_assesment_dataset_10k_cleaned.csv"

# Ensure the directory exists
data_dir.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(data_dir / "normalization_assesment_dataset_10k.csv")

# Preprocess the text column
df = preprocessor.preprocess_column(df, 'raw_comp_writers_text')

# Check if any of the files exist
if not all(path.exists() for path in [examples_path, test_path, full_path]):
    # Split and save the data only if the files do not exist
    preprocessor.split_and_save_data(df, examples_path, test_path, full_path)

# Load examples
examples = TextNormalizer.load_examples(examples_path)

# Initialize the TextNormalizer with examples
llm_normalizer = TextNormalizer(openai_api_key=OPENAI_API_KEY, examples=examples)

# Load spaCy model for NER
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic", message=".*torch.utils._pytree._register_pytree_node.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch", message=".*torch.cuda.amp.autocast.*")
    nlp = spacy.load("en_pipeline")

class TextRequest(BaseModel):
    text: str

@app.post("/normalize_llm")
async def normalize_llm(request: TextRequest):
    query = request.text
    query = preprocessor.preprocess_text(query)
    if query:
        response = llm_normalizer.normalize_text(query)
        final_response = ""
        for normalized_text in response.normalized_text:
            #join with new line
            final_response += f"{normalized_text.CLEAN_TEXT}\n"
        return {"normalized_text": final_response}
    else:
        raise HTTPException(status_code=400, detail="Empty input text")

@app.post("/normalize_ner")
async def normalize_ner(request: TextRequest):
    dataset_prep = DatasetPrep()
    cleaned_text = dataset_prep.preprocess_text(request.text)
    print(cleaned_text)
    doc = nlp(cleaned_text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    if entities:
        joined_entities = "/".join(entities)
        return {"normalized_text": joined_entities}
    else:
        return {"normalized_text": ""}