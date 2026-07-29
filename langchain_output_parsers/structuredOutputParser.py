# StructuredOutputParser ye result ko json data me convert kr deta hai 
# or ye json output parser se isliye alg h kyuki ye json data ke sath sath schema bhi provide krta hai lekin ye strictly typed ni hota


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser
# from langchain_core.output_parsers import ResponseSchema
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv(Path(__file__).resolve().parents[1] /'.env', override=True)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name ='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name ='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name ='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='give three fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)



chain = template | model | parser

result = chain.invoke({'topic':'blackhole'})

print(result)