import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_List =[
            'Delhi is the capital of India',
            'Lion is the national animal of India',
            'IPL is a indian cricket league'
                ]
        return {'response': random.choice(response_List)}
    
    def predict(self, prompt):
        
        response_List =[
            'Delhi is the capital of India',
            'Lion is the national animal of India',
            'IPL is a indian cricket league'
        ]
        return {'response': random.choice(response_List)}
#humne iss NakliLLM class ko child bna diya abstract class ka jisme invoke function hai or ab inn child class ke liye ye jruri ho gya ki invoke function inko bhi bnana hi pde ga or jse hi ye bna toh humne predict class ka sara logic invoke me dal diya or ab .invoke krke hum result le skte h 
llm = NakliLLM()
result = llm.invoke("What is the capital of India")
# result2 = llm.predict("What is the capital of India")
# print(result)
# print(result2)

class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
        
    def invoke(self, input_dict):
        return self.template.format(**input_dict) #**input_dict = **{'topic':'india', 'length':'short'} → unpack hoke ban jata hai topic='india', length='short'
    
    
    def format(self, input_dict):
        return self.template.format(**input_dict)

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['topic', 'length']
)

prompt = template.invoke({'topic':'india','length':'short'})
# result = llm.invoke(prompt)
# print(result)


class RunnableConnector(Runnable):
    
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
        
    def invoke(self, input_data):
        
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data
chain = RunnableConnector([template, llm])

result = chain.invoke({'topic':'india','length':'short'})

print(result)

        

