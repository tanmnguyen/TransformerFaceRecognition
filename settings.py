import os 
import time 
import torch 

from utils.io import read_config

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

class Settings:
    def __init__(self, config_path=None):
        if config_path:
            self.config_dict = read_config(config_path)
            self._set_config()
    
    def _set_config(self):
        # set device config 
        self.config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set save path config 
        self.config_dict['result_path'] = os.path.join(
            "results", 
            f"{self.config_dict['arch']}-{get_time()}"
        )

    def update_config(self, config_path):
        new_config = read_config(config_path)
        self.config_dict.update(new_config)
        self._set_config()

    def __getattr__(self, name):
        if name in self.config_dict:
            return self.config_dict[name]
        return None 
            
# set default config 
settings = Settings("configs/default.cfg")
