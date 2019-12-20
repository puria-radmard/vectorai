# In this file, I've taken the map embedded on this page: https://directories.lloydslist.com/ 
# and taken all the markers listed on its associated .json file to a list

# On the page source, the map can be found at .html address: 
# /html/body/div/div[2]/div[2]/div/div[2]/div/div[3]/div/div/div/form/div[2]/div/script/text()

# From here, we learn that the marker information on their maps is from the JSON
# https://directories.lloydslist.com/script/stockist_map_details.json

# Now we can parse the JSON for the name of every marker

import json
from urllib.request import urlopen

url = "https://directories.lloydslist.com/script/stockist_map_details.json"
response = urlopen(url).read()
data = json.loads(response.decode('utf-8'))
#print(str(data)[:100])                                        # The JSON was weirdly split in two, and the terminal only displayed the last portion of the source, so I had to inspect what the opening keyword was

outlist = []
for place in data["places"]:
    outlist.append(str(place["title"]))

# For the final generation, I'll throw in country names too

import pandas as pd
import place_scraper

place_scraper.scrapeWikiPage("https://en.wikipedia.org/wiki/List_of_sovereign_states", [0], outlist)

d = {"data": outlist, "type": ["Location" for x in outlist]}
outcsv = pd.DataFrame(data = d)
outcsv.to_csv("C:/Users/puria/source/repos/puria-radmard/vectorai/Latest Data/locations.csv")