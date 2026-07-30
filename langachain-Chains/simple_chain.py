from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate

load_dotenv(Path(__file__).resolve().parents[1] /'.env', override=True)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

parser = StrOutputParser()

template = PromptTemplate(
    template='Give me 5 intreseting facts about {topic}',
    input_variables=['topic']
)

model = ChatHuggingFace(llm = llm)


Chain = template | model | parser

result = Chain.invoke({'topic':'blackhole'})


print(result)

