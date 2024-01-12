#%% Mixtral 3_K_M
#Simple text completion# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
from llama_cpp import Llama

llm = Llama(
  model_path="mixtral-8x7b-v0.1.Q3_K_M/mixtral-8x7b-v0.1.Q3_K_M.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
)

question = "What is a mongoose (animal)? Do they make good pets? Would they harm my cat?"
output = llm(
    "Q: {question} A: ".format(question=question), # Prompt
    max_tokens=3000, # Generate up to 32 tokens
    stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

#%% Mixtral 4_K_M
#Chat Completion API
from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(model_path="mixtral-8x7b-v0.1.Q4_K_M/mixtral-8x7b-v0.1.Q4_K_M.gguf",
            chat_format="llama-2",  # Set chat_format according to the model you are using
            n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=28         # The number of layers to offload to GPU, if you have GPU acceleration available
)

llm.create_chat_completion(
    messages = [
        #{"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Would a mongoose make a good pet if I keep rats?"
        }
    ]
)

#%% Mixtral 4_0
#Chat Completion API
from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(model_path="mixtral-8x7b-v0.1.Q4_0/mixtral-8x7b-v0.1.Q4_0.gguf",
            chat_format="llama-2",  # Set chat_format according to the model you are using
            n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=28         # The number of layers to offload to GPU, if you have GPU acceleration available
)

llm.create_chat_completion(
    messages = [
        #{"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Would a mongoose make a good pet if I keep rats?"
        }
    ]
)

#%% Nous Capybara 5_K_S
#Chat completion api
from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(model_path="nous-capybara-34b.Q5_K_S/nous-capybara-34b.Q5_K_S.gguf",
            chat_format="llama-2",  # Set chat_format according to the model you are using
            n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=28         # The number of layers to offload to GPU, if you have GPU acceleration available
)

# llm.create_chat_completion(
#     messages = [
#         {"role": "system", "content": "You are a story writing assistant."},
#         {
#             "role": "user",
#             "content": "Would a mongoose make a good pet if I keep rats?"
#         }
#     ]
# )

question = "What is a mongoose (animal)? Do they make good pets? Would they harm my cat?"
output = llm(
    "Q: {question} A: ".format(question=question), # Prompt
    max_tokens=3000, # Generate up to 32 tokens
    stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

#%%Try Langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 59  # Change this value based on your model and your GPU VRAM pool.
n_batch = 20000  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="nous-capybara-34b.Q5_K_S/nous-capybara-34b.Q5_K_S.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

#ask question
question = "Write a best man speech for my friend Jordan"

#read in the context for the question
with open('dataset.txt') as f:
    context = f.readlines()

# #Template format 1:
# template = """Question: {question}
# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# #Template format 2
# template = f"<s>[INST] Using this information : {context} \
#                     answer the Question : {question} [/INST]\n"
    
#Template format 3
template = """USER: {question} ASSISTANT:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(question)

# %%
