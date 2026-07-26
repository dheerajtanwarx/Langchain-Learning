# The model receives multiple messages, not just one string.
# for example
# System:
# You are a helpful assistant.

# Human:
# Explain Machine Learning in simple words.

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert')
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'virat kohli'})

print(prompt)