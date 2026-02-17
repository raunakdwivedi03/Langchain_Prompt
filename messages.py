from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

model=ChatHuggingFace()

messages=[
    SystemMessage(content='You are ta helpful assistent'),
    HumanMessage(content='Tell me about langchain')
    
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

