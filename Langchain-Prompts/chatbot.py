from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


# env_path = Path(__file__).resolve().parents[1] / ".env"

# print("Current file:", Path(__file__).resolve())
# print("Looking for .env at:", env_path)
# print(".env exists:", env_path.exists())

# load_dotenv(env_path, override=True)



model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

chat_history = [
    SystemMessage(content = 'You are a helpful AI assistant')
] 

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)