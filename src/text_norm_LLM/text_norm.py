from typing import List, Dict
import pandas as pd
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from text_norm_LLM.schemas import NormalizedTextResponse




class TextNormalizer:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o", examples: List[Dict[str, str]] = None):
        self.openai_api_key = openai_api_key
        self.model = model
        self.llm = self._initialize_llm()

        # Load examples if provided
        self.examples = examples if examples else []
        
        # Initialize example selector once
        self.example_selector = self._initialize_example_selector()

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize and cache the LLM to avoid repeated initialization."""
        return ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key, model=self.model)

    def _initialize_example_selector(self) -> SemanticSimilarityExampleSelector:
        """Initialize the example selector once and reuse it."""
        return SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            Chroma,
            k=5,
        )

    @staticmethod
    def load_examples(file_path: str) -> List[Dict[str, str]]:
        """Load examples from a CSV file and cache the result."""
        examples = pd.read_csv(file_path).to_dict(orient="records")
        for example in examples:
            for key, value in example.items():
                if isinstance(value, float):
                    example[key] = str(value)
        return examples

    def _create_prompt_template(self) -> FewShotPromptTemplate:
        """Create a FewShotPromptTemplate for text normalization using the preloaded example selector."""
        prompt_example = PromptTemplate.from_template("""
            Raw text to normalize:
            {raw_comp_writers_text}

            Normalized text:
            {CLEAN_TEXT}
        """)

        return FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=prompt_example,
            input_variables=["query"],
            prefix="""
            You are an expert in the music industry. 
            Your task is to normalize writer names by removing redundant information.
            Below are some examples of raw composer/writer names and their normalized versions:
            Note: 
            - dont include the same name twice in the normalized version
            - Account for non-latin characters
            """,
            suffix="""Normalize the following:
            {query}""",
        )

    def normalize_text(self, query: str) -> NormalizedTextResponse:
        """Normalize the input query using the cached example selector."""
        prompt_template = self._create_prompt_template()
        structured_llm = self.llm.with_structured_output(NormalizedTextResponse)

        print(prompt_template.invoke({"query": query}).to_string())
        chain = prompt_template | structured_llm

        return chain.invoke({"query": query})
