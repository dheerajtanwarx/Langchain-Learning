# File Read Karta Hai: Yeh local text file ko specified encoding (jaise utf-8) ke sath open karta hai.Document Object Banata Hai: Raw text ko LangChain ke Document object mein wrap (convert) kar deta hai.

from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)

model = AzureChatOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

prompt = PromptTemplate(
    template='give a small summary about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

loader = TextLoader('animal.txt', encoding='utf-8')

docs = loader.load() #loader ko load krdo docs me 

chain = prompt | model | parser

result = chain.invoke({'topic':docs[0].page_content})

print(result)

# print(type(docs))
# print(docs) file ka sbb kuch dikh jaye ga 
#print(docs[0].page_content)  # File ka poora text dikhega
#print(docs[0].metadata)      # Output: {'source': 'your_file.txt'}

