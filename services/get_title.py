from llm_factory.get_llm import get_hf_llm
import re


def get_chat_title(model_name, user_query):
    llm = get_hf_llm(selected_model=model_name)

    prompt = f"""
You are a helpful assistant that generates short, clear, and catchy titles.
Task: - Read the given user query.
- Create a concise title (max 7 words, no extra text).
User Query: {user_query}
Output: Title:
"""
    # Generate the response
    title_response = llm.generate(prompt, max_tokens=20)

    # 1️⃣ Extract raw text depending on the response type
    if hasattr(title_response, "text"):
        title_text = title_response.text
    elif isinstance(title_response, list):
        title_text = title_response[0].get('generated_text', '')
    else:
        title_text = str(title_response)

    # 2️⃣ Remove quotes, newlines, "Title:" prefix, extra spaces
    title_text = title_text.strip().strip('"').strip("'")
    if title_text.lower().startswith("title:"):
        title_text = title_text[6:].strip()
    title_text = title_text.replace("\n", " ").strip()

    # 3️⃣ Keep only the first 7 words
    words = title_text.split()
    title_text = " ".join(words[:7])

    return title_text

# Example usage
# model_name = "TinyLlama"
# user_query = "Can you explain the concept of reinforcement learning and its applications in modern AI?"
# title = get_chat_title(model_name, user_query)
# print(title)