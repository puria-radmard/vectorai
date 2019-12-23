# This file aims to scrape for names of vessels from:
# https://www.marinetraffic.com/en/data/?asset_type=vessels&columns=flag,shipname,photo,recognized_next_port,reported_eta,reported_destination,current_port,imo,ship_type,show_on_live_map,time_of_latest_position,lat_of_latest_position,lon_of_latest_position

# To make it easier for myself, I reduced the number of columsn by going to:
# https://www.marinetraffic.com/en/data/?asset_type=vessels&columns=shipname
# which gives just the one column table of ship names

# (I tried to just click export data at the top, but this cost money)

# The issue with this page is that the link does not change when you click next page, so a web driver like Selenium has to be used

# I then realised that the page given only provides 20*25 = 500 ships, so I instead used:
# https://www.fleetmon.com/vessels/
# for scraping

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import time


def scrapeWithScript(url, xpath, iteratorID):

    outlist = []

    driver = webdriver.Chrome("C:\\Users\\puria\\.wdm\\drivers\\chromedriver\\79.0.3945.36\\win32\\chromedriver.exe")
    driver.get(url)
    n = 0

    TIMEOUT  = 5
    LOADING_ELEMENT_XPATH = "vesselsearch_table_processing"

    while len(outlist) <= 300:
        for elem in driver.find_elements_by_xpath("/html/body/main/div/div/table/tbody/tr/td/div/div/strong/a"):
            if elem.text not in outlist: outlist.append(elem.text)
            n+=1
        
        try:

            elem = driver.find_element_by_id(iteratorID)
            ActionChains(driver).move_to_element(elem).click(elem).perform()
            
            WebDriverWait(driver, TIMEOUT).until(EC.presence_of_element_located((By.XPATH, LOADING_ELEMENT_XPATH)))
            WebDriverWait(driver, TIMEOUT).until_not(EC.presence_of_element_located((By.XPATH, LOADING_ELEMENT_XPATH)))

        except:                 # Sometimes you get hit with an advert, and you can just click back page and resume to the same page
            driver.get(url)

    driver.quit()
    return outlist

data = scrapeWithScript(url = "https://www.fleetmon.com/vessels/",
                 xpath =  "/html/body/main/div/div/table/tbody/tr/td/div/div/strong/a",
                 iteratorID = "vesselsearch_table_next")

import pandas as pd

d = {"data": data, "type": ["Vessel" for x in data]}
outcsv = pd.DataFrame(data = d)
outcsv.to_csv("C:/Users/puria/source/repos/puria-radmard/vectorai/Latest Data/vessels.csv")