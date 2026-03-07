from config.settings import Settings

settings = Settings()


def get_hf_models_list():
    """
    Returns a list of available HuggingFace models
    based on HF_MODEL_NAME dictionary in settings.
    """
    hf_models = list(settings.HF_MODEL_NAME.keys())
    return hf_models


# Example usage
# check_hf_models = get_hf_models_list()
# print(type(check_hf_models))   # <class 'list'>
# print(check_hf_models)         # ['TinyLlama', 'Phi-2']