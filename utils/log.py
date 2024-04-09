import os 
from settings import settings

def log(message):
    log_path = os.path.join(settings.result_path, "log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'a') as f:
        f.write(f"{message}\n")
        print(message)