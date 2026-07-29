from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser 
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name , age and city of a fictional person \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}#iska mtlb hai "Return a JSON object. Do not return any extra text. Wrap your keys in double quotes..." ye run time se phle hi chl jata hai or llm ko instruction mil jati h ki result kse bhejna hai 
)

#without using chains
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# print(prompt) #this is use to print the format of the template 
# print(final_result['name'])
# print(type(final_result))


#by using chains
chain = template | model | parser
result = chain.invoke()
print(result)