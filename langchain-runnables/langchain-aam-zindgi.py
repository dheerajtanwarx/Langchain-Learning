import random


class NakliLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        
        response_List =[
            'Delhi is the capital of India',
            'Lion is the national animal of India',
            'IPL is a indian cricket league'
        ]
        return {'response': random.choice(response_List)}
    
llm = NakliLLM()
# result = llm.predict("What is the capital of India")
# print(result)

class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def format(self, input_dict):
        return self.template.format(**input_dict)

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['topic', 'length']
)

prompt = template.format({'topic':'india','length':'short'})
# result = llm.predict(prompt)
# print(result)

class NakliLLMChain:
    
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
        
    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        
        return result['response']
    
# template2 = NakliPromptTemplate(
#     template='Write a {length} poem about {topic}',
#     input_variables=['topic', 'length']
# )

# llm2 = NakliLLM()

chain = NakliLLMChain(llm, template)

result=chain.run({'length':'short', 'topic':'india'})
print(result)