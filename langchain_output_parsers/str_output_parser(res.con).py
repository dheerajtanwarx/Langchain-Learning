# this is the method by using result.content basically hu, dekh rhe hai ki result.content or stroutput parser me ky difference hai


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import os
from langchain_core.prompts import PromptTemplate

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)

llm = HuggingFaceEndpoint(
     repo_id="meta-llama/Llama-3.1-8B-Instruct",
      task="text-generation",
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm = llm)

#prompt template 1 => report
template1 = PromptTemplate(
    template='write a detailed report on. \n {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'blackhole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content })

result1 = model.invoke(prompt2)

print(result1.content)