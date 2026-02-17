import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=api_token,
    temperature=0.7
)

chat_history=[
    SystemMessage(content= 'you are a helpful Ai assistent')
]


model = ChatHuggingFace(llm=llm)

print("AI Chat Started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)