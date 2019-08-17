import json
import pandas as pd
from pandas.io.json import json_normalize
from pandas.io.json.normalize import nested_to_record


items = []
with open('training_tiny.jsonl') as data_file:
    for line in data_file:
        # print(line)
        json_data = json.loads(str(line).strip())
        data = nested_to_record(json_data, "features")
        items.append(data)

frame = pd.DataFrame(data=items)
print(frame)


