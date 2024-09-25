import os
import dotenv

dotenv.load_dotenv()


def get_target_model_name():
    return os.getenv('TARGET_MODEL_NAME', 'gpt-4o-mini')


def get_eval_model_name():
    return os.getenv('EVAL_MODEL_NAME', 'gpt-4o-mini')


def get_opro_model_name():
    return os.getenv('OPRO_MODEL_NAME', 'gpt-4o')


def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')
