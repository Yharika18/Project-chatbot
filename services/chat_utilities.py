from llm_factory.get_llm import get_hf_llm

def format_chat_prompt(chat_history):

    system_prompt = "You are a helpful AI assistant. Answer the user's question clearly and concisely.\n\n"

    conversation = ""

    for msg in chat_history:
        role = msg["role"].lower()

        if role == "user":
            conversation += f"User: {msg['content']}\n"

        elif role == "assistant":
            conversation += f"Assistant: {msg['content']}\n"

    conversation += "Assistant:"

    return system_prompt + conversation


def get_answer(model_name, chat_history):
    llm = get_hf_llm(selected_model=model_name)

    prompt = format_chat_prompt(chat_history)

    response = llm.generate(prompt, max_tokens=200)

    return response
#example usage
model_name = "TinyLlama"

chat_history = [
    {"role": "user", "content": "What is Artificial Intelligence?"}
]

response = get_answer(model_name, chat_history)
print(response)