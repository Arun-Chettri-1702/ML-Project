import logging 
import os
from datetime import datetime

LOG_FILE =f"{datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log"
logspath =os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(logspath,exist_ok=True)
log_file_path =os.path.join(logspath,LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
