# Import required modules
import sys
import json
import uuid
import logging
import traceback
from asyncio import Queue
from config import settings
from typing import Annotated
from redis.client import Redis
from supabase import create_client
from pydantic import BaseModel, constr


from langchain.cache import RedisCache
from langchain.schema import Generation
from langchain.chains import RetrievalQA
from redis import BlockingConnectionPool
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.messages import SystemMessage
from langchain.schema.runnable import RunnableConfig

from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, Tool
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler

from langchain.memory import (
    MotorheadMemory,
    VectorStoreRetrieverMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
    ConversationBufferWindowMemory,
)
# Initialize global variables
REDIS_CHAT_URL = settings.redis_chat_url
REDIS_CACHE_HOST = settings.redis_cache_host
REDIS_CACHE_PORT = settings.redis_cache_port
REDIS_CACHE_PW = settings.redis_cache_pw 
SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_key
OPENAI_API_KEY = settings.openai_api_key

# Initialize global clients
SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
EMBEDDING_CLIENT = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Initialize  redis clients
cache_conn_pool = Redis(connection_pool=BlockingConnectionPool(host=REDIS_CACHE_HOST, port=REDIS_CACHE_PORT, password=REDIS_CACHE_PW, max_connections=2, timeout=20))
redis_cache = RedisCache(cache_conn_pool, ttl=604800)
llm_string = "gpt-3.5-turbo"

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = configure_logging()  
session_id=str(uuid.uuid4())
message_queue = Queue(maxsize=2) 
manager = AsyncCallbackManager([])
run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(callbacks=[run_collector])

def get_session():
    return RedisChatMessageHistory(session_id=str(uuid.uuid4()), ttl=None)

# Create the constrained string type
ConstrainedStr = constr(min_length=2, max_length=1000, to_lower=True, strip_whitespace=True)

class ChatInput(BaseModel):
    text: Annotated[str, constr(min_length=2, max_length=1000, to_lower=True, strip_whitespace=True)]
    
@app.middleware("http")
async def add_session(request: Request, call_next):
    response = await call_next(request)
    return response

@app.get("/chat")
async def chat(input_message: str): 
    await message_queue.put(input_message)
    input_message = await message_queue.get()
    message_history = RedisChatMessageHistory(
        url=REDIS_CHAT_URL, session_id=session_id, ttl=600, 
    ) 
    # Define the system message
    system_message = SystemMessage(
        content=
        "You are a friendly and helpful chatbot for a website called hopecommunities.org." 
        "Your goal is to first help users who have questions about the website and second to encourage them to visit the website and maybe even support what the organization is doing by donating and/or getting involved as a volunteer and so on."
        "Answer all questions using terms like 'our' and 'we' to reinforce ownership and emulate 'agent_on_behalf' responsibility." 
        "Assume all questions are about the website and give special attention to users saying they speak a different language than english to quickly match and provide them with the information on the HopeCommunity Team member who speaks the language."
    )
    # Create the prompt
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
    )
    llm = ChatOpenAI(verbose=True, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
    memory = ConversationTokenBufferMemory(chat_memory=message_history, memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=7000)
    #memory = ConversationSummaryBufferMemory(chat_memory=message_history, memory_key="chat_history", return_messages=True, llm=streaming_llm, max_token_limit=9000)
    #memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
    # Initialize vector store and memory
    vectorstore = SupabaseVectorStore (client=SUPABASE_CLIENT, embedding=EMBEDDING_CLIENT, table_name='documents', query_name='match_documents')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    tools = [
            Tool(name='KnowledgeBase',
                func=qa.run,
                description='use this tool when answering all questions to retrieve relevant documents for user query then provide the most relevant and useful answer as a chatbot for a website(Hope Communities) whose knowledge store you have access to. Input should be a brief and precise query using the least amount of words possible for example; User: "are there events?". Tool Input: "Hopecommunity events". User: "i speak farsi". Tool Input: "team farsi" '),
        ] 
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        callback_manager=manager,
        verbose=True,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )
    # Check cache
    cached_resp = redis_cache.lookup(input_message, llm_string)
    if cached_resp:
        #message_history.add_message(AIMessage(content=cached_resp[0].text, type="ai"))
        return JSONResponse(content={"response": {"output": cached_resp[0].text}})
    # Run the conversation and generate output
    output = await agent_executor.arun((
                {
                    "input": input_message,
                }
            ))
    # Update cache
    return_val = [Generation(text=output)]
    redis_cache.update(input_message, llm_string, return_val) 
    return JSONResponse(content={"response": {"output": output}})
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
        traceback_str = traceback.format_exc()
        logger.error(f"Error occurred while handling the query: {str(exc)}\n{traceback_str}")
        return JSONResponse(status_code=500, content={"message": "An error occurred. Please try again later."})
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chatbot API!"}


from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory="templates")

# Mount a static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/landing_page")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
