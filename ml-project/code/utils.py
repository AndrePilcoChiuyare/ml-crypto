import json

# load JSON
def loadJSON(filepath):
    with open(filepath) as file:
        return json.load(file)