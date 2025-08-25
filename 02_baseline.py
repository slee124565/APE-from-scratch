import asyncio
import pandas as pd
from prompt_evaluator import PromptEvaluator
from google.genai.types import HarmCategory, HarmBlockThreshold
from google.genai import types

if __name__ == "__main__":
    df_train = pd.read_csv('test.csv')  # Load your training data

    target_model_name = "gemini-2.5-flash-lite"
    target_model_config = {
        "temperature": 0, "max_output_tokens": 5000
    }
    review_model_name = "gemini-2.5-flash-lite"
    review_model_config = {
        "temperature": 0, "max_output_tokens": 100
    }
    # safety_settings = {
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    # }
    safety_settings = [
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        )
    ]
    review_prompt_template_path = 'review_prompt_template.txt'  # Path to the review prompt text file

    evaluator = PromptEvaluator(
        df_train, target_model_name, target_model_config,
        review_model_name, review_model_config, safety_settings,
        review_prompt_template_path
    )

    default_prompt = 'Solve the given problem about geometric shapes.'
    prompt = input("Please enter the prompt for evaluation: ")
    if not prompt:
        prompt = default_prompt

    asyncio.run(evaluator.main(prompt))
