# Imports --------------------------------------------------------------------------------------------------
# https://huggingface.co/tasks/table-question-answering

from transformers import pipeline
import pandas as pd
import time
import math
import os

# Small snippet to retrieve coordinates from a geojson file
import requests
import json


#Working Directory setzen
os.chdir('C:/Users/Jens_/Documents/Unterlagen/Studium Dresden/2. Semester/GIT06/GID-Project')


# Beispielpolygone von Auriol ----------------------------------------------------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/aurioldegbelo/geod2023/main/8_poly_examples/bundeslaender_poly_examples.geojson"
res = requests.get(url)
features = json.loads(json.dumps(res.json()['features']))

coord_bayern = features[0]['geometry']['coordinates']
coord_berlin = features[1]['geometry']['coordinates']
coord_saxony = features[2]['geometry']['coordinates']


# Testing the snippet on polygons -----------------------------------------------------------------------------------------------
data = {"City": ["Dresden", "Berlin", "Munich"],
        "Point-Geometry": ["[51.049259, 13.73836]", "[52.518611, 13.408333]", "[48.139722, 11.574444]"],
        "Polygon-Geometry": [str(coord_saxony), str(coord_bayern), str(coord_berlin)]}
table = pd.DataFrame.from_dict(data)
print(table)

question = "What is the polygon geometry of Dresden?"

# pipeline model
# Note: you must to install torch-scatter first.
# result with original table
t = round(time.time())
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
print(tqa(table=table, query=question)['cells'][0])
t = round(time.time())-t
print(str(math.floor(t) // 3600) + "::" + str((t-(math.floor(t) // 360)*360) // 60) + "::" + str((t-(math.floor(t) // 60)*60))) # ~4-5min



# Etwas optimieren -----------------------------------------------------------------------------------------
data = {"City": ["Dresden", "Berlin", "Munich"],
        "Point-Geometry": ["[51.049259, 13.73836]", "[52.518611, 13.408333]", "[48.139722, 11.574444]"],
        "Polygon-Geometry": [str(coord_saxony), str(coord_bayern), str(coord_berlin)]}
table = pd.DataFrame.from_dict(data)
question = "What is the polygon geometry of Dresden?"

# ich will die Koordinatenzeile in einem neuen Vektor abspeichern un die KZ durch den Zeilenindex ersetzen
geom = table["Polygon-Geometry"]
table["Polygon-Geometry"] = [str(i) for i in range(table.shape[0])]
print(table)

# Ergebnis
t = round(time.time())
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
i = int(tqa(table=table, query=question)['cells'][0])
print(geom[i])
t = round(time.time())-t
print(str(math.floor(t) // 3600) + "::" + str((t-(math.floor(t) // 360)*360) // 60) + "::" + str((t-(math.floor(t) // 60)*60))) # ~1min



# GADM Deutschland Beispielpolygone --------------------------------------------------------------------
# Vorbereitung
# Daten laden
with open('Daten/GeoJSON/gadm41_DEU_1.json', 'r', encoding='utf-8') as json_datei:
    daten = json.load(json_datei)
    os.close('Daten/GeoJSON/gadm41_DEU_1.json')
table = pd.DataFrame.from_dict(daten["features"])


# Daten in passendes Format bringen (Pandas Data-Frame)
prop = table['properties'] #Properties in die Tabelle integrieren
prop = pd.DataFrame(list(prop))
del table['properties']
table = pd.concat([table, prop], axis = 1)

geom = table["geometry"] # Koordinatenzeilen abspeichern und durch Indexe ersetzen (Berechnungsdauer)
table["geometry"] = [str(i) for i in range(table.shape[0])]
print(table)


#Frage
question = "what is the geometry of Sachsen?"


# Berechnung
t = round(time.time()) # Berechnungszeit messen
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
i = tqa(table=table, query=question)['cells'][0]
try:
  i = int(i)
  answer = geom[i]
except:
  InterruptedError ('Falsche Spalte (Es wurde kein Index ausgegeben)')
  answer = 'answer: ' + i

t = round(time.time())-t
t = str(math.floor(t) // 3600) + "::" + str((t-(math.floor(t) // 360)*360) // 60) + "::" + str((t-(math.floor(t) // 60)*60))


# Überprüfung
print(answer)
table.iloc[i]
print(t) # ~1min




