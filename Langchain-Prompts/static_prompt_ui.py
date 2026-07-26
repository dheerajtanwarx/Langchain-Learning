# this is static prompt ui yaha par prompt ka pura control user ke pas hota h pura promt user likhta h or uske hisab se usko result mil jata hai 
# A static prompt is a fixed prompt where nothing changes. You write the entire prompt yourself.
# For ex: Explain Artificial Intelligence in simple words.
#If tomorrow you want to ask about Machine Learning, you must manually change the code.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
import os
import streamlit as st

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

user_input = st.text_input('Enter Your Prompt')

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)

