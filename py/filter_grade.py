import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, default='../data/2016+2017.json', help='filename of input JSON data')
parser.add_argument('--output_json', type=str, default='output.json', help='filename of output JSON data')
parser.add_argument('--grade', type=str, default='7A', help='selected grade')
opt = parser.parse_args()
print(opt)

with open(opt.input_json) as input_file:
    json_data = json.load(input_file)

filtered_json = list(filter(lambda x : x['Grade'] == opt.grade, json_data))

with open(opt.output_json, 'w') as outfile:
    json.dump(filtered_json, outfile, indent=1)
