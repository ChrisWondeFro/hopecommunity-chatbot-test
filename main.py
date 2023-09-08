# Import required modules
import sys
import json
import uuid
import logging
import traceback

from asyncio import Queue
from pydantic import constr
from config import settings
from redis.client import Redis
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client

from contextvars import ContextVar
from langchain.agents import AgentType
from langchain.cache import RedisCache
from langchain.chains import RetrievalQA
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import Tool, initialize_agent
from redis import BlockingConnectionPool, RedisError
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AIMessage, HumanMessage, Generation
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict, _message_to_dict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

# Initialize global variables
REDIS_CHAT_HOST = settings.redis_chat_host
REDIS_CHAT_PORT = settings.redis_chat_port 
REDIS_CHAT_PW = settings.redis_chat_pw

REDIS_CACHE_HOST = settings.redis_cache_host
REDIS_CACHE_PORT = settings.redis_cache_port
REDIS_CACHE_PW = settings.redis_cache_pw 

OPENAI_API_KEY = settings.openai_api_key
SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_key

# Initialize global clients
SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
EMBEDDING_CLIENT = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# Initialize  redis clients
cache_conn_pool = Redis(connection_pool=BlockingConnectionPool(host=REDIS_CACHE_HOST, port=REDIS_CACHE_PORT, password=REDIS_CACHE_PW, max_connections=2, timeout=16))
redis_cache = RedisCache(cache_conn_pool)

chat_conn_pool = Redis(connection_pool=BlockingConnectionPool(host=REDIS_CHAT_HOST, port=REDIS_CHAT_PORT, password=REDIS_CHAT_PW, max_connections=2, timeout=18))
redis_chat_client = chat_conn_pool

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

# Context variable for RedisChatMessageHistory
session_var = ContextVar("session_var", default=None)

message_queue = Queue(maxsize=1)

last_output = None 

def get_session():
    return RedisChatMessageHistory(session_id=str(uuid.uuid4()), ttl=2400)

class ChatInput(BaseModel):
    text: constr(min_length=2, max_length=1000, to_lower=True, strip_whitespace=True)

class RedisChatMessageHistory:
    def __init__(self, session_id: Optional[str] = None, ttl: Optional[int] = None):
        self.session_id = session_id if session_id else str(uuid.uuid4())
        self.ttl = ttl
        self.key_prefix = "message_store:"
        
    def key(self) -> str:
        return f"{self.key_prefix}{self.session_id}"
    
    def messages(self) -> List[BaseMessage]:
        try:
            _items = redis_chat_client.lrange(self.key, 0, -1)
            items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
            return messages_from_dict(items)
           
        except RedisError:
            print("RedisError occurred while reading messages.")
            return []   

    def add_message(self, message: BaseMessage) -> None:
        if not isinstance(message, BaseMessage):
            raise ValueError(f"Expected a BaseMessage instance, got {type(message)} instead")
    
        if not hasattr(message, 'type'):
            raise ValueError("The message object lacks a 'type' attribute")
        try:
            serialized_message = messages_to_dict(messages=[message])
            redis_chat_client.lpush(self.key(), json.dumps(serialized_message))
            if self.ttl:
               redis_chat_client.expire(self.key(), self.ttl)
             
        except RedisError:
            print("RedisError occurred while adding message.")       
            
    def clear(self) -> None:
        redis_chat_client.delete(self.key)

def init_chat_session():
    return RedisChatMessageHistory(session_id=str(uuid.uuid4()), ttl=2400)

# Function for creating prompt messages
def create_prompt_messages(input_message, latest_messages):
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(
            "You are a friendly and helpful AI chatbot for a website called Hope Communities website. Answer questions using 'our' and 'we' when providing information about the website using the tools."
        ),
    ]
    for message in latest_messages:
        if message == "human":
            prompt_messages.append(HumanMessagePromptTemplate.from_template(input_message, SystemMessagePromptTemplate))
        else:
            prompt_messages.append(AIMessagePromptTemplate.from_template(message))
    prompt_messages.append(HumanMessagePromptTemplate.from_template("{question}"))
    return prompt_messages

@app.middleware("http")
async def add_session(request: Request, call_next):
    session = init_chat_session()
    session_var.set(session)
    response = await call_next(request)
    return response

@app.get("/chat")
async def chat(input_message: str, session: RedisChatMessageHistory = Depends(get_session)):
    global last_output 
    await message_queue.put(input_message)
    input_message = await message_queue.get()
    
    # Initialize key components
    message_history = session
    latest_messages=message_history.messages.__str__()

    prompt_messages = create_prompt_messages(input_message, latest_messages)
    prompt_messages.append(HumanMessagePromptTemplate.from_template("{question}")) 

    llm = ChatOpenAI(verbose=True, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
    convers_memory = ConversationBufferWindowMemory(human_prefix='human', ai_prefix='AI', memory_key="chat_history", return_messages=True, k=1)
    # Initialize vector store and memory
    vectorstore = SupabaseVectorStore (client=SUPABASE_CLIENT, embedding=EMBEDDING_CLIENT, table_name='hope_communities', query_name='match_documents')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    tools = [
            Tool(name='Knowledge Base', func=qa.run, description='use this tool when answering all questions to retrieve relevant documents for user query then provide the most relevant and useful piece of information, answer as a chatbot for a website(Hope Communities) whose knowledge-store you have acess to'),
        ]
    agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=convers_memory, prompt_messages=prompt_messages) 
 
    # Sample input message (in this case it comes from the FastAPI endpoint)
    message_history.add_message(HumanMessage(content=input_message, type="human"))  # Save input to Redis

    # Check cache
    cached_resp = redis_cache.lookup(input_message, llm_string)
    if cached_resp:
        message_history.add_message(AIMessage(content=cached_resp[0].text, type="ai"))
        return JSONResponse(content={"response": {"output": cached_resp[0].text}})
    
    # Run the conversation and generate output
    output = agent_executor.run(input_message)
    message_history.add_message(AIMessage(content=output, type="ai"))
    
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