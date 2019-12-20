# First let's build a scraper that scans certain Wikipedia pages for place names, testing first on a list of countries

import requests
from bs4 import BeautifulSoup

def scrapeWikiPage(url, cols, outlist):

    # Convert the url into a readable
    res = requests.get(url).text
    soup = BeautifulSoup(res)

    for col in cols:
        for j in soup.find_all('table', class_= "wikitable"):    # Find all tables
            for i in j.find_all(["tr"]):                         # Iterate through table bodies
                
                name = None

                row = i.find_all(['th', 'td'])                   # Find each row
                try:
                    name = row[col].find_all("a")[0].text.rstrip()        # Add the first hyperlink from the [col] cell onto our list
                    if name in outlist: name = None                       # .rstrip() removes trailing line breaks

                except: pass

                try:
                    name = row[col].text.rstrip()                # For rows without hyperlinks
                    if name in outlist: name = None             # Early removal
                
                except: pass
                

                if name != None:       
                    finname = name.split("(")[0]            # Removes annotation like "(ancient kingdom)"
                    outlist.append(finname)                 # Actually adds to the list it was input, rather than create new one
    
pages = [("https://en.wikipedia.org/wiki/List_of_sovereign_states", [0]),
 ("https://en.wikipedia.org/wiki/List_of_oldest_continuously_inhabited_cities", (0,1,2)),
 ("https://en.wikipedia.org/wiki/List_of_cities_or_metropolitan_areas_by_GDP", (0,1,2)),
 ("https://en.wikipedia.org/wiki/List_of_cities_in_Iran_by_province", [1])]

finlist = []

for page in pages:
    scrapeWikiPage(page[0], page[1], finlist)   

print(finlist)