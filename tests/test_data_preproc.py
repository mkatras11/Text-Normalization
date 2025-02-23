import pytest
import pandas as pd
from src.text_norm_LLM.data_preproc import TextPreprocessor

@pytest.fixture
def sample_data():
    data = {
        'text_column': [
            'This is a sample text.',
            '<Unknown> This should be cleaned.',
            'Traditional song',
            'Unknown writer',
            'Some random text',
            '///Leading and trailing delimiters///',
            '    White spaces    ',
            'Valid text'
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_preprocess_text(preprocessor):
    assert preprocessor.preprocess_text('<Unknown>/This should be cleaned.') == 'This should be cleaned.'
    assert preprocessor.preprocess_text('/Leading and trailing delimiters/') == 'Leading and trailing delimiters'
    assert preprocessor.preprocess_text(' White spaces ') == 'White spaces'
    assert preprocessor.preprocess_text('Traditional song') == 'song'
    assert preprocessor.preprocess_text('Some random text') == 'Some random text'
    assert preprocessor.preprocess_text('') == ''
    assert preprocessor.preprocess_text('Valid text') == 'Valid text'