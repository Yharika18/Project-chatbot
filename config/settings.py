from dotenv import load_dotenv
from pydantic_settings import BaseSettings
load_dotenv() # loading the values into environment variables
# From env variables it takes all values and store in variables like MONGO_DB_URL,Mongo_db_name and later it can use
class Settings(BaseSettings):
     MONGO_DB_URL: str
     MONGO_DB_NAME: str
     DEFAULT_MODEL: str  = "TinyLlama"
     HF_MODEL_NAME: dict = {
         "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         "Phi-2": "microsoft/phi-2"
     }

     # config class is used if setting class variables(MONGO_DB_URL,etc) are not available it automatically loads the variables of config from env file
     class Config:
         env_file = ".env"
         env_file_encoding = "utf-8"