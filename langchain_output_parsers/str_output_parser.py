# Sabse phle StrOutputParser ka sbse common use hota hai chain ke sath  
#StrOutputParser : ye ek llm ko force karta hai ki wo apna output ek string format me bheje  

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

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

# ye hum mostly isliye use krte hai kyu ye bhout simple syntex hai as compare to .content ,,ke reference ke liye  aap str_output_parser(res.con) wali file dekh skte hai 
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
#humne ek chain bnayi usme sbse phle aaya temp1 wo gya model ke pas phir jo uska result mila usko humne parse kiya mtlb usme se useful chij uthai jo ki ek string thi  jaise phle hum result.content krte the phir wo humne temp 2 ko bheja or phir wo wapis gya model me or uska 5 line summary wala result generate hua or usko parse kr liya or print kr diya 
result = chain.invoke({'topic':'blackhole'})

print(result)