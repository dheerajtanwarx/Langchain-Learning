# Import the Azure OpenAI chat model integration wrapper from LangChain's OpenAI partner package
from langchain_openai import AzureChatOpenAI
# Import the load_dotenv function to read configuration variables from a local text-based environment file
from dotenv import load_dotenv
# Import the standard string parser that extracts clean text output from raw model message objects
from langchain_core.output_parsers import StrOutputParser
# Import Python's built-in operating system interface module to fetch system environment variables
import os
# Import the Path class from Python's standard library to construct and manage absolute file system paths
from pathlib import Path
# Import the PromptTemplate class used for dynamically injecting runtime user variables into static text blocks
from langchain_core.prompts import PromptTemplate
# Import LangChain Expression Language (LCEL) core components used to create parallel, conditional, or custom steps
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
# Import the parser that forces LLMs to output matching JSON schemas and turns JSON strings into Python objects
from langchain_core.output_parsers import PydanticOutputParser
# Import BaseModel to define custom schemas and Field to describe data variables inside Pydantic structures
from pydantic import BaseModel, Field
# Import Literal to define strict validation choices where a variable can only equal specific fixed strings
from typing import Literal

# Locate, resolve, and load your .env file from exactly one folder level higher than this current running file
load_dotenv(Path(__file__).resolve().parents[1] /'.env', override=True)

# Instantiate the Azure OpenAI chat client wrapper using your loaded deployment, endpoint, and version configurations
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Instantiate a string output parser (Note: This specific instance 'parser' is initialized but never chained below)
parser = StrOutputParser()

# Create a data schema model named Feedback to define the exact JSON output format we expect from the LLM
class Feedback(BaseModel):
    # Enforce that the sentiment key must be a string containing exactly the value 'positive' or 'negative'
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Instantiate a Pydantic parser hooked to your Feedback class to automatically generate prompt formatting rules
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Define prompt1 to feed the text and inject structural JSON formatting parameters into the main prompt text
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)
# Define prompt2 to construct a positive response prompt template that strictly expects a variable named 'feedback'
prompt2 = PromptTemplate(
    template='write a appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback'],

)
# Define prompt3 to construct a negative response prompt template that strictly expects a variable named 'feedback'
prompt3 = PromptTemplate(
    template='write a appropriate response to this Negative feedback \n {feedback}',
    input_variables=['feedback'],

)

# Construct a conditional routing branch that evaluates the data structure it receives from the upstream step
branch_chain = RunnableBranch(
    # Route 1: Check if the input object's sentiment attribute is 'positive'. If True, run this sub-chain pipeline
    (lambda x : x.sentiment == 'positive', prompt2 | model | parser2),
    # Route 2: Check if the input object's sentiment attribute is 'negative'. If True, run this sub-chain pipeline
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser2),
    # Default Route: Executes a custom lambda fallback node returning text if all previous conditions evaluate to False
    RunnableLambda(lambda x: "Could not find sentiment")

)

# Combine components into a standalone classifier pipeline: Prompt Template -> Azure Model -> Pydantic Structure Parser
classifier_chain = prompt1 | model | parser2

# Assemble your final chain piping the output from classifier_chain directly as input into your branch_chain
chain = classifier_chain | branch_chain

# Execute the complete connected chain sequence synchronously by passing your initial user feedback text dictionary
result = chain.invoke({'feedback':'this phone is beautiful'})
# Print the final execution result directly to your terminal window console output screen
print(result)

# Execute the standalone classifier chain directly with negative text to retrieve only its specific sentiment value
# result = classifier_chain.invoke({'feedback':'This phone is worst'}).sentiment
# Print the extracted isolated string sentiment value directly to your terminal window console output screen
# print(result)
