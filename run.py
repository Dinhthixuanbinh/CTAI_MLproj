import yaml
import argparse
from src.main import main

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = get_config()
    main(config)