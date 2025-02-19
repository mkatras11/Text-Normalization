"Prompts for the text normalization task"


PROMPT_TEMP = """
            Raw text to normalize:
            {raw_comp_writers_text}

            Normalized text:
            {CLEAN_TEXT}
        """

FEW_SHOT_PRE = """
            You are an expert in the music industry. 
            Your task is to normalize writer names by removing redundant information.
            Below are some examples of raw composer/writer names and their normalized versions:
            Note: 
            - dont include the same name twice in the normalized version
            - Account for non-latin characters
            """

FEW_SHOT_SUF = """Normalize the following:
            {query}"""