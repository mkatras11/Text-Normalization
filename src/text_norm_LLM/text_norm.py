" A module to normalize text data using a structured output model llm with semantic few-shot learning. "
from typing import List, Dict
import pandas as pd
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from text_norm_LLM.schemas import NormalizedTextResponse
from text_norm_LLM.prompts import PROMPT_TEMP, FEW_SHOT_PRE, FEW_SHOT_SUF



class TextNormalizer:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o", examples: List[Dict[str, str]] = None):
        self.openai_api_key = openai_api_key
        self.model = model
        self.llm = self._initialize_llm()

        # Load examples if provided
        self.examples = examples if examples else []
        
        # Initialize example selector
        self.example_selector = self._initialize_example_selector()

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM to avoid repeated initialization."""
        return ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key, model=self.model)

    def _initialize_example_selector(self) -> SemanticSimilarityExampleSelector:
        """Initialize the example selector."""
        return SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            Chroma,
            k=5,
        )

    @staticmethod
    def load_examples(file_path: str) -> List[Dict[str, str]]:
        """Load examples from a CSV file in the format of a list of dictionaries."""
        examples = pd.read_csv(file_path).to_dict(orient="records")
        for example in examples:
            for key, value in example.items():
                if isinstance(value, float):
                    example[key] = str(value)
        return examples

    def _create_prompt_template(self) -> FewShotPromptTemplate:
        """Create a FewShotPromptTemplate for text normalization using the preloaded example selector."""
        prompt_example = PromptTemplate.from_template(PROMPT_TEMP)

        return FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=prompt_example,
            input_variables=["query"],
            prefix=FEW_SHOT_PRE,
            suffix=FEW_SHOT_SUF,
        )

    def normalize_text(self, query: str) -> NormalizedTextResponse:
        """Normalize the input query using the example selector."""
        prompt_template = self._create_prompt_template()
        structured_llm = self.llm.with_structured_output(NormalizedTextResponse)

        chain = prompt_template | structured_llm

        return chain.invoke({"query": query})
