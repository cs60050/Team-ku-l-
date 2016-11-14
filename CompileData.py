### this file combines the individual JSON files scrapped of the website and writes it into a CSV file
### Some prelimany cleaning like joing all sentences and removing '\n\r' chars is also done in this file


import json
import csv
from pprint import pprint
from collections import defaultdict

### Opening each JSON file, storing its content and closing it

with open('etEdits.json') as json_data:
    et = json.load(json_data)
    json_data.close()

with open('dcComment.json') as json_data:
    dc = json.load(json_data)
    json_data.close()

with open('fexEdits.json') as json_data:
    fex = json.load(json_data)
    json_data.close()

with open('iexpressEdits.json') as json_data:
    iex = json.load(json_data)
    json_data.close()

with open('toiEdits.json') as json_data:
    toi = json.load(json_data)
    json_data.close()

with open('gaurdianEdits.json') as json_data:
    guardian = json.load(json_data)
    json_data.close()


### Extracting only the Editorials from the entire dataset.
### Dataset also contained information like date, author and tags - information which is not relevant for trainng and testing

etEdits = [result['ContentParagraph'] for result in et]
dcComment = [result['ContentParagraph'] for result in dc]
fexEdits = [result['ContentParagraph'] for result in fex]
iexEdits = [result['ContentParagraph'] for result in iex]
toiEdits = [result['ContentParagraph'] for result in toi]
guardianEdits = [result['ContentParagraph'] for result in guardian]


### Intialising a dictionary to store values
dictionary = defaultdict(list)

### Writing values to the dictionary
### Also some preliminary data cleaning is done while Writing
### Basic cleaning such as replacing \n and \r
### Also since it is an article it contains values like comma and tabs etc.
### So, we wanted to use '*' as a delimiter
### So instances of asterisks were replaced with '#' to enable that.

for edits in etEdits:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("Economic Times")

for edits in dcComment:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("Deccan Chronicle")

for edits in fexEdits:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("Financial Express")

for edits in iexEdits:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("Indian Express")

for edits in toiEdits:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("TOI")

for edits in guardianEdits:
    oneEdit = ""
    for sentences in edits:
        oneEdit += str(sentences) + " "
    oneEdit = oneEdit.replace('\n', ' ').replace('\r', '').replace('*', '#')
    dictionary[oneEdit].append("Guardian")


### Writing the output to a csv file delimited by '*' because of reasons mentioned in the previous comment

with open('dict2.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter='*')
    i = 0
    for key, value in dictionary.items():
       writer.writerow([i, key, value])
       i +=1
