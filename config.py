from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Chatbot API"
    
    openai_api_key: str
    supabase_url: str
    supabase_key: str

    redis_chat_host: str
    redis_chat_port : int
    redis_chat_pw : str
    redis_chat_url:str

    redis_cache_host: str
    redis_cache_port : int
    redis_cache_pw : str
    
    class Config:
        env_file = ".env"

settings = Settings()
