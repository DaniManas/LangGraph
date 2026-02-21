from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI( model="Qwen/Qwen2.5-7B-Instruct:together", api_key=os.environ["HF_TOKEN"], base_url="https://router.huggingface.co/v1" )

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

def chat_node(State: ChatState):
    messages = State['messages']
    response = llm.invoke(messages)
    return {'messages':[response]}


graph.add_node('chat_node',chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

