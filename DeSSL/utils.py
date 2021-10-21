# -*- coding: UTF-8 -*-
import argparse
import yaml
import ast

__all__ = ['loadding_config']

def loadding_config(cfg_path : str) -> args.ArgumentParser:
    '''
    load a config from Yaml file.

    Args:
        cfg_path: The path of yaml file.

    Returns:
        An ArgumentParser that describes hyper parameters.
    '''
    with open(cfg_path, 'r') as fin:
        cfg = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
        cfg = [x for x in cfg][0]
    parser = argparse.ArgumentParser()
    for item in cfg:
        if isinstance(cfg[item], bool):
            parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item])
        else:
            parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item])
    return parser