import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(var_name, default_value=None, var_type=str):
    return var_type(os.getenv(var_name, default_value))
