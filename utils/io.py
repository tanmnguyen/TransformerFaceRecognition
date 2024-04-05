import os 
import configparser

def get_absolute_path(file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
    return file_path

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(get_absolute_path(config_path))
    
    return dict(config.items('DEFAULT'))