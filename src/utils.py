import yaml
import logging

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(name)s: %(message)s'
    )
    return logging.getLogger(name)