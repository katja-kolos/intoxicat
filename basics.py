import json


def read_json(filename):

    with open(filename, 'r') as jsn:
        file_contents = json.load(jsn)

    return file_contents