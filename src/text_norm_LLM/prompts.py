"Prompts for the text normalization task"


PROMPT_TEMP = """
            Example raw text:
            {raw_comp_writers_text}

            Example normalized text:
            {CLEAN_TEXT}
        """

FEW_SHOT_PRE = """
            You are an expert in the music industry. 
            Your task is to normalize writer names by removing redundant information.
            
            Note:
            - Only include the name of the composer/writer in the normalized version. No publishers, dates, or other information.
            - Dont include the same name twice in the normalized version.
            - Identify and include non-latin characters in the normalized version.
            - Every letter capital or lower in the raw text should be capital or lower in the normalized text.

            Below are some examples of raw composer/writer names and their normalized versions:
            """

FEW_SHOT_SUF = """Normalize the following:
            {query}"""