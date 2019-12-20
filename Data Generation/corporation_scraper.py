# Helpfully, https://opencorporates.com/ splits its companies per register on https://opencorporates.com/registers
# First I will scrape one of these, then I'll scrape all the links for there from the main page, and scrape each one

# If you go too far down you get some dodgy wesbite names that just waren't reflective of the rest of the data, so for each 
# register I will sort by relevance and limit to first 1200 (the smallest linked register, Abu Dhabi's, has 1297 companies)

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import time

def scrapeRegister(url, driver):

    outlist = []

    driver.get(url)

    # Sort by relevance
    driver.find_element_by_link_text('Sort by relevance').click()
    
    for i in range(3):

        for elem in driver.find_elements_by_xpath("/html/body/div/div/div/div/ul/li/a"):
                if elem.text not in outlist: outlist.append(elem.text)

        next_button = driver.find_element_by_xpath("/html/body/div[2]/div[2]/div[1]/div[2]/div/div[1]/ul/li[8]/a")
        next_button.click()

    return outlist

# After testing this function, I realised the website has an automatic blocker after 3 pages. What might be a good idea is to
# scrape 3 pages on each register then come back to them, going times times over to make 900 companies per register

#scrapeRegister("https://opencorporates.com/companies/ae_az")

def findRegisters(url):

    outlist = []
    pages = []

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)

    for elem in driver.find_elements_by_xpath("/html/body/div/div/div/div/div/div/table/tbody/tr/td/a[contains(@href, '/companies/')]"):
        new_page = elem.get_attribute("href")
        pages.append(new_page)
        
    for page in pages[0:2]:
        print(page)
        outlist += scrapeRegister(page, driver)
        #print(scrapeRegister(link, driver))
    
    return outlist

data = findRegisters("https://opencorporates.com/registers")

# Now to log these in a .csv

import pandas as pd

d = {"data": data, "type": ["Company" for x in data]}
outcsv = pd.DataFrame(data = d)
outcsv.to_csv("C:/Users/puria/source/repos/puria-radmard/vectorai/Latest Data/companies.csv")