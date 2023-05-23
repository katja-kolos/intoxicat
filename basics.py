import json


def read_json(filename):

    with open(filename, 'r') as jsn:
        file_contents = jsn.load()

    return file_contents