import os 
from settings import settings

log_path = os.path.join(settings.result_path, "log.txt")
def log(message, log_path=log_path):
    with open(log_path, 'a') as f:
        f.write(f"{message}\n")
        print(message)