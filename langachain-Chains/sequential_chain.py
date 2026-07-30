from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='give a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

Chain = template1 | model | parser | template2 | model | parser

result = Chain.invoke({'topic':'Unemployment in India'})

print(result)

