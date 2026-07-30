from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
#  Modern correct import
from langchain_core.runnables import RunnableParallel

load_dotenv(Path(__file__).resolve().parents[1] /'.env', override=True)


# inki limit puri ho gyi
# llm1 = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task='text-generation',
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    
# )
# model1 = ChatHuggingFace(llm=llm1)

model1 = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
model2 = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

prompt1 = PromptTemplate(
    template='Generate simple and short notes from the following text \n, {text}',
    input_variable=['text']
)

prompt2 = PromptTemplate(
    template='Generate Five short question answers from the following text  \n {text} ',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document  \n notes => {notes} and quiz => {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser,
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """An AI Agent is an autonomous entity that uses a Large Language Model as its central decision-making engine.
Unlike static linear chains, an agent does not follow a hardcoded sequence of steps to solve a problem.
The primary role of the LLM inside an agent architecture is to determine which actions to take and in what order.
Agents use a loop called the ReAct framework, which stands for Reasoning and Acting, to solve complex user prompts.
In the reasoning phase, the agent evaluates the current state and writes out a thought process about what to do next.
In the action phase, the agent selects a specific tool from its available toolkit and generates its input parameters.
Tools are interfaces that allow an agent to interact with the outside world, databases, APIs, or local code sandboxes.
A tool consists of three essential elements: a name, a detailed textual description, and an argument schema.
The description of a tool is critical because the LLM reads it to decide if that tool is appropriate for the task.
If a tool description is poorly written or vague, the agent may suffer from tool invocation failures or hallucinations.
LangChain provides built-in tools for common tasks, such as Tavily Search, Wikipedia API, ArXiv paper search, and Python REPL.
Custom tools can be created easily by applying the tool decorator directly above a standard Python function.
When using the tool decorator, LangChain automatically extracts the function's docstring to use as the tool description.
Type hints inside a decorated function are used by LangChain to automatically generate the tool's input validation schema.
The input variables required by tools are strictly validated at runtime to prevent malformed data from executing.
An Agent Executor is the runtime environment that manages the loop of thinking, selecting tools, and executing actions.
The Agent Executor calls the chosen tool, captures the output observation, and feeds it back to the LLM.
This loop repeats iteratively until the LLM determines that it has enough information to provide a final response.
To prevent infinite loops, the Agent Executor can be configured with a maximum iteration limit parameter.
Another safety feature is the max execution time limit, which cuts off the agent if a tool hangs or takes too long.
The transition from legacy agents to LangGraph represents the modern standard for building complex agentic systems.
LangGraph treats agents as state machines, where steps are represented as nodes and transitions are edges.
A state graph allows developers to define cyclical workflows, which are difficult to manage in standard LCEL chains.
The state object in LangGraph is a shared data structure that gets updated progressively by every node in the graph.
Nodes in a state graph are standard Python functions that accept the current state and return updated state keys.
Edges determine the control flow, directing the system from one node to the next based on conditions or paths.
A conditional edge uses a router function to evaluate the current state and dynamically choose the next node.
An example of a conditional edge is checking if a tool's output contains an error, then routing to a correction node.
Memory can be attached to agents using a checkpointer, which saves the state graph's history after every single step.
This persistent state enables time travel features, allowing developers to replay, debug, or fork an agent's history.
ChatHuggingFace models can drive agents effectively if they have been fine-tuned for tool calling capabilities.
Tool calling models output structured data, like JSON, identifying the exact name and arguments of the tool to run.
Models without native tool calling must rely on prompt engineering frameworks like JSON formatting or XML tags.
The structural template used for non-tool-calling models is often referred to as the Structured Chat Agent pattern.
When an agent misinterprets tool outputs, it can get stuck in a repetitive loop, calling the same tool with the same input.
To mitigate tool loops, developers can inject specific system prompt instructions guiding the agent on failure recovery.
Human-in-the-loop patterns allow an agent to pause execution and wait for manual human approval before running a tool.
Human approval is highly recommended for sensitive tool actions, such as sending emails, deleting data, or making financial trades.
The compile method on a LangGraph state graph transforms the abstract design into a runnable component.
Once compiled, the graph inherits all standard Runnable methods like invoke, stream, batch, and ainvoke.
Streaming from an agent can return either graph state updates or the raw token stream directly from the underlying LLM.
Multi-agent systems consist of multiple independent specialized agents that collaborate to solve a large, complex task.
In a multi-agent architecture, a supervisor agent often routes sub-tasks to subordinate worker agents.
Worker agents can communicate with each other by passing messages through a shared central state graph.
This modularity ensures that individual agents remain highly accurate because their toolkit is small and specialized.
An agent with too many tools in its toolkit often suffers from degradation in accuracy due to prompt context dilution.
The input to an agent graph is typically a dictionary containing a list of messages representing the conversation history.
The final output of an agent loop is an intermediate or final message that satisfies the user's original objective.
Debugging agents is significantly easier using LangSmith, which visualizes every thought, tool call, and tool response.
LangSmith traces reveal exactly what the LLM saw right before it made an incorrect tool routing decision.
Type safety within agents ensures that tool outputs are correctly converted to strings before being read by the model.
Legacy agents like InitializeAgent are now deprecated in favor of explicit graph-based definitions.
Modern agent design patterns separate the planning model from the execution models to save computational costs.
For instance, a smaller open-source model can execute simple math tools, while a larger model handles the main planning.
Autonomy in agents ranges from strictly guided router pipelines to completely open-ended goal-driven explorers.
Strictly guided agents use predefined graph pathways, ensuring high predictability in commercial enterprise applications.
Open-ended agents have more freedom to choose their paths, making them useful for creative research and discovery tasks.
A major challenge in agent deployment is handling non-deterministic behavior, where the same input yields different paths.
Testing agents requires evaluating historical trace datasets rather than comparing simple string outputs.
Enterprise agent frameworks integrate security boundaries to ensure tools cannot access unauthorized system files.
The ultimate goal of Agentic AI is to transition from passive question-answering systems to active digital workers.
Mastering agents and state-based architectures is the final step in building truly intelligent AI applications.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()