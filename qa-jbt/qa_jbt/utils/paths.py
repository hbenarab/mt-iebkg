__author__ = 'bernhard'

import os
import shutil


def create_paths(data_path, temp_path, paths=list()):
    """
    Cleans the output path to remove files form previous runs.
    """
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    if not os.path.isdir(data_path):
        os.mkdir(data_path, mode=0o770)
    os.mkdir(temp_path, mode=0o770)
    for p in paths:
        os.mkdir(p, mode=0o770)
