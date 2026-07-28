# In LangChain, a message placeholder (MessagesPlaceholder) is a tool used in chat prompt templates to dynamically insert a whole list of messages—such as past conversation history—at a specific spot during runtime


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []
#load chat history

with open ('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

#create prompt

prompt = chat_template.invoke({'chat_history':chat_history, 'query':'where is my refund'})

print(prompt)