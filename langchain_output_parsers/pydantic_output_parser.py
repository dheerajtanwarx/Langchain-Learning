from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] /'.env', override=True)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    # Return full text restriction lagana zaroori hai backend formatting ke liye
    return_full_text=False #return_full_text=False: Yeh open-source HuggingFace model ko force karta hai ki purana raw prompt text output mein include mat karein. Iske bina model poora prompt bapas repeat karta hai jiski vajah se Pydantic core library identify nahi kar paati ki string kahan se start ho rahi hai.
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(description='age of the person')
    city: str = Field(description='Name of the city  the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name , age ,city of  the fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# chain = template | model | parser

# result = chain.invoke({'place':'india'})

# print(parsed_result)
 
 
prompt = template.invoke({'place':'india'})
result = model.invoke(prompt)
parsed_result = parser.parse(result.content)
# print(prompt)
print(parsed_result)

    
