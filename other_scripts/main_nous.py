
#%%
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = CTransformers(model="TheBloke/Nous-Capybara-34B-GGUF")
template = """Question: {question} Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run("What is AI?")

#%%
response = llm_chain.run("How were you made?")
print(response)