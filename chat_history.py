from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

import yaml
import os

output_parser = StrOutputParser()

## Ollama 
model = OllamaLLM(model="llama3.2", temperature=0.0)

ai_response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print(ai_response)

ai_response = model.invoke([HumanMessage(content="What's my name?")])
print(ai_response)

ai_response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

print(ai_response)

## OpenAI
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

ai_response = model.invoke([HumanMessage(content="Hi! I'm Bob")])

output_parser.invoke(ai_response)

ai_response = model.invoke([HumanMessage(content="What's my name?")])
output_parser.invoke(ai_response)

ai_response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

output_parser.invoke(ai_response)


