#%% Import libs
# to access the local fo|lder for the models use: cd ~/.cache/huggingface/hub/
import torch
import pickle 

# Device info
torch_mem_info = torch.cuda.mem_get_info()
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Globally available:', round(torch_mem_info[0]/1024**3,1), 'GB')
    print('Total:   ', round(torch_mem_info[1]/1024**3,1), 'GB')

# Check GPU compatibility with bfloat16 (pre ampere GPUs probably won't be able to use it)
compute_dtype = getattr(torch, 'float16')
if compute_dtype == torch.float16 and True:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#%% Example 1 Mistral-7B-OpenOrca-GPTQ: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ
# #Revision1: gptq-8bit-32g-actorder_True
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-8bit-32g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="gptq-8bit-32g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#%% Example 2 Mistral-7B-Instruct-v0.2-GPTQ: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
#Revision1: gptq-8bit-32g-actorder_True
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-8bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-8bit-32g-actorder_True")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#%% Example 3 TheBloke/Solar-10.7B-SLERP-GPTQ: https://huggingface.co/TheBloke/Solar-10.7B-SLERP-GPTQ
# #Revision1: gptq-8bit-32g-actorder_True
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/Solar-10.7B-SLERP-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-8bit-32g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="gptq-8bit-32g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#%% Example 4 TheBloke/Solar-10.7B-Instruct-v1.0-GPTQ: https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GPTQ
# #Revision1: gptq-8bit-32g-actorder_True
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/SOLAR-10.7B-Instruct-v1.0-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-8bit-32g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="gptq-8bit-32g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

#%% Example 5 Nous-Capybara-34B-GPTQ: https://huggingface.co/TheBloke/Nous-Capybara-34B-GPTQ
# # Revision1: main
# # Revision2: gptq-3bit-32g-actorder_True
# # Revision3: gptq-3bit-128g-actorder_True

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/Nous-Capybara-34B-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-4bit-128g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=True,
#                                              revision="gptq-3bit-128g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)

#%% Example 6 Nous-Hermes-2-Yi-34B-GPTQ: https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GPTQ
# # Revision1: main
# # Revision2: gptq-3bit-128g-actorder_True

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/Nous-Hermes-2-Yi-34B-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-3bit-128g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=True,
#                                              revision="gptq-3bit-128g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)

#%% Example 7 Mixtral-8x7B-v0.1-GPTQ:gptq-3bit-128g-actorder_True : https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GPTQ
# # Revision1: gptq-3bit--1g-actorder_True
# # Revision2: gptq-3bit-128g-actorder_True

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_name_or_path = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" # or TheBloke/Mixtral-8x7B-v0.1-GPTQ
# # To use a different branch, change revision
# # For example: revision="gptq-3bit-128g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=True,
#                                              revision="gptq-3bit--1g-actorder_True")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
#%% Try a simple prompt
# prompt = "Write a story about llamas"
# system_message = "You are a writing assistant"
# prompt_template=f'''{prompt}
# '''
# print("\n\n*** Generate:")
# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))

#%% Set question answer prompt
#Read in the dataset
# with open('dataset4.txt') as f:
#     context = f.readlines()

# prompt = 'How are companies using RAG with their LLMs?'
# system_message = "You are a coding assistant" # only use this for mixtral
# prompt_template = f"<s>[INST] Using this information : {context} \
#                     answer the Question : {prompt} [/INST]\n"

# print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, 
#                         top_p=0.95, top_k=40, max_new_tokens=5000)
# print(tokenizer.decode(output[0]))
# result = tokenizer.decode(output[0])

# text_file = open("output.txt", "w")
# text_file.write(result)
# text_file.close()

#%% Create pipeline for LangChain
# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=5000,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     return_full_text=True,
#     repetition_penalty=1.1
# )
# llm_output = pipe(prompt_template)
# print(llm_output[0]['generated_text'])

#%% Using Langchain to create the LLM Chain
from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
# import nest_asyncio
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

#%% Now langchain example
#Read in the context dataset
with open('dataset4.txt') as f:
    context = f.readlines()

#Turn the context document into a langchain.document.object
context_doc = []
for document in context:
    context_doc.append(Document(page_content=document, metadata={"source": "local"}))

#chunk the doc for vector transformation
text_splitter = CharacterTextSplitter(chunk_size=100, 
                                      chunk_overlap=0)
chunked_context_doc = text_splitter.split_documents(context_doc)
# Create vector embeddings database
# Load chunked documents into the FAISS index
vector_context_db = FAISS.from_documents(chunked_context_doc, 
                                         HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
# Connect query to FAISS index using a retriever
vector_db_retriever = vector_context_db.as_retriever(search_type="similarity",
                                                     search_kwargs={'k': 4})

# Try out the db with a query
prompt = "What is RAG?"
db_docs = vector_context_db.similarity_search(prompt)
print(db_docs[0].page_content)

#Set the prompt template
prompt_template = "<s>[INST] Using this information : {context} \
                    answer the Question : {prompt} [/INST]\n"

#Create the HF pipeline
print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    return_full_text=True,
    repetition_penalty=1.1
)

#Build the HF pipeline into the chain
mistral_llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "prompt"],
    template=prompt_template,
)

# Create llm chain 
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

#%% Create vector database by using Langchain FAISS API online tutorial: https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
# nest_asyncio.apply()
# articles = ["https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/",
#             "https://www.fantasypros.com/2023/11/5-stats-to-know-before-setting-your-fantasy-lineup-week-10/",
#             "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
#             "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
#             "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/"]
# # Scrapes the blogs above
# loader = AsyncChromiumLoader(articles)
# docs = loader.load()

# # Converts HTML to plain text 
# html2text = Html2TextTransformer()
# docs_transformed = html2text.transform_documents(docs)

# # Chunk text
# text_splitter = CharacterTextSplitter(chunk_size=100, 
#                                       chunk_overlap=0)
# chunked_documents = text_splitter.split_documents(docs_transformed)

# # Create vector embeddings database
# # Load chunked documents into the FAISS index
# db = FAISS.from_documents(chunked_documents, 
#                           HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
# # Dump the vector DB to disk for future use
# file = open('vector_database.pickle', 'wb')
# pickle.dump(db, file)
# file.close()

# # Connect query to FAISS index using a retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 4}
# )

# # Just read from disk once you do previous code chunk
# file = open('vector_database.pickle', 'rb')
# db = pickle.load(file)
# file.close()

# # Connect query to FAISS index using a retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 4}
# )

# # Try out the db with a query
# prompt = "What did Laporta say?"
# docs = db.similarity_search(prompt)
# print(docs[0].page_content)

#%% Generate gemeric LLM response to test
llm_output = llm_chain.invoke({"context": '', 
                                 "prompt":"Should I use RAG for work with LLMs?"})
print(llm_output['text'])

#%% Create a RAG chain that can query embeddings
prompt = "Should I use RAG for work with LLMs? Answer in point form." 
retriever = vector_context_db.as_retriever()

rag_chain = ( 
 {"context": retriever, "prompt": RunnablePassthrough()}
    | llm_chain
)
chain_output = rag_chain.invoke(prompt)
print(chain_output['text'])

#%% Now we need to check to see if this would work on a database for querying
# load in the DB
bldg_db = SQLDatabase.from_uri("sqlite:///chinook.db")

# Create agent executor for database 
agent_executor = create_sql_agent(
    llm=mistral_llm,
    toolkit=SQLDatabaseToolkit(db=bldg_db, llm=mistral_llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

#%% Example 1: run queries
agent_executor.run(
    "is there a table about customers?"
)

#%% Generate SQL queries
db_chain = SQLDatabaseChain.from_llm(mistral_llm, bldg_db, verbose=True)
tables_df = db_chain.run("Who are the artists recorded in the artist table of the database?")

# %%
