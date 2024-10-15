from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage
)


chat = ChatOpenAI(
    api_key="YOUR API KEY",
    base_url="http://localhost:6006/v1",
    temperature=0,
    model_name='Qwen2.5-7B-Instruct'
)

message = [HumanMessage(content="你是谁?")]

print(chat(message))