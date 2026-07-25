from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt #load prompt ka use hum external file se prompt ko load krne ke liye krte h

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)

llm = HuggingFaceEndpoint(
     repo_id="meta-llama/Llama-3.1-8B-Instruct",
     task='text-generation',
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
 
st.header("Research Tool")


paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical","Code-Oriented", "Mathematical"] )


length_input = st. selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation) "])

# humare pass ek method toh ye tha ki hum iss prompt ko yhi likh kr use kr le or dusra ye hai ek koi dusri file me prompt likh kar waha se koi bhi file me load kr le . best case is create another file i.e prompt_template.py and load here
# template = PromptTemplate(
# template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}
# Explanation Length: {length_input}
# 1. Mathematical Details:
# - Include relevant mathematical equations if present in the paper.
# - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
# 2. Analogies:
# - Use relatable analogies to simplify complex ideas.
# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=['paper_input', 'style_input', 'length_input']
# )

#ye hai external file se prompt ko laod krne ka tarika
template = load_prompt('template.json')

# fill the placeholders
prompt = template.invoke(
    {
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    }
)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
