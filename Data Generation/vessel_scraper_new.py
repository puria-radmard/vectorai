# The fleetmon scraper would time out after a while due to mass request limits
# Tried one with vesselfinder.com since they have a url change ==> easier to make requests

# 20 ships per page so we will try out 300 pages to get 6000 ships

import selenium
from selenium import webdriver
import pandas as pd

def scrapeVesselPages():

    outlist = []
    driver = webdriver.Chrome("C:\\Users\\puria\\.wdm\\drivers\\chromedriver\\79.0.3945.36\\win32\\chromedriver.exe")

    url_mt = 'https://www.vesselfinder.com/vessels?page={}'

    for i in range(300):

        url = url_mt.format(str(i+1))
        driver.get(url)

        try:

            for elem in driver.find_elements_by_xpath("/html/body/div/div/main/div/section/table/tbody/tr/td/a"):
                if elem.text not in outlist and not "": outlist.append(elem.text)
            
            print(len(outlist))

        except:

            break

    return outlist[1:]
    
data = scrapeVesselPages()

d = {"data": data, "type": ["Vessel" for x in data]}
outcsv = pd.DataFrame(data = d)
outcsv.to_csv("C:/Users/puria/source/repos/puria-radmard/vectorai/Latest Data/vessels.csv")