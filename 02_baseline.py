import asyncio
import dotenv
import pandas as pd
import config
from prompt_evaluator import PromptEvaluator

# from vertexai.generative_models import HarmBlockThreshold, HarmCategory

dotenv.load_dotenv()

if __name__ == "__main__":
    df_train = pd.read_csv('test.csv')  # Load your training data

    target_model_name = config.get_target_model_name()
    target_model_config = {
        "temperature": 0, "max_output_tokens": 1000
    }
    review_model_name = config.get_eval_model_name()
    review_model_config = {
        "temperature": 0, "max_output_tokens": 10
    }
    safety_settings = {
        # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        # HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    review_prompt_template_path = 'review_prompt_template.txt'  # Path to the review prompt text file

    evaluator = PromptEvaluator(
        df_train, target_model_name, target_model_config, review_model_name, review_model_config, safety_settings,
        review_prompt_template_path
    )

    prompt = input("Please enter the prompt for evaluation: ")
    # prompt = 'Solve the given problem about geometric shapes.'
    asyncio.run(evaluator.main(prompt))
