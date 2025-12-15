import configparser
import os
import platform
import sys

config = None

def get_config(section, key):
    global config
    if config is not None:
        return config.get(section, key)

    config = configparser.ConfigParser()
    local_config_path = os.path.normpath('./config.ini')
    if os.path.exists(local_config_path):
        config_path = local_config_path
    else:
        if platform.system() == 'Linux':
            config_path = '/etc/datu/dtai/config.ini'
        else:
            raise Exception(f"Unsupported operating system: {platform.system()}")
    # print("config_path: ", config_path)

    if os.path.exists(config_path):
        config.read(config_path)
        return config.get(section, key)
    else:
        raise FileNotFoundError("Configuration file not found.")