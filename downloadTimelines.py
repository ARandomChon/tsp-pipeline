# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 09:14:51 2021

@author: saipr
"""
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
import time

WebLink = "https://steamdb.info/graph/"

#initialize the chrome webpage that will gather the information
driver = webdriver.Chrome(executable_path='C:/Users/saipr/Desktop/chromedriver.exe')
driver.get(WebLink)

#pause for a minute so that we know the page is loaded
time.sleep(5)

#now we have a page loaded. First things first, make sure we're showing 1K pages
select = Select(driver.find_element_by_name('table-apps_length')) #find the options tag
select.select_by_value('1000') #select what number of values we want, 25 for testing, 5000 for the scrape

#pause for a minute so that we know the page is loaded
time.sleep(5)

#now we write the file to HTML to keep, so we can open it in BS4 and do the rest
file = open('steamGamesList.html', 'w', encoding='utf8')
file.write(driver.page_source)
file.close()

driver.close()

#now open the HTML file in BS4
content = open('steamGamesList.html','r', encoding='utf8')
soup = BeautifulSoup(content)

#find all the trs with class = app and put them into a list
linksList = []
games = soup.findAll("tr", {"class": "app"})
for g in games:
    aa = g.find("a", href=True)
    link = str("https://steamdb.info" + aa['href'])
    print(link)
    linksList.append(link)
    
#and now print and export this list
dat = pd.DataFrame(linksList)
dat.to_csv('gamesList.csv')

#with each of these links, we need to download their timeline files
#options = Options()
#options.page_load_strategy = 'normal'
#options.add_argument("start-maximized")
#options.add_experimental_option("excludeSwitches", ["enable-automation"])
#options.add_argument("--enable-javascript")
#for l in linksList:
    #print(l)
    #driver2 = webdriver.Chrome(options=options, executable_path='C:/Users/saipr/Desktop/chromedriver.exe')
    #driver2.get(l)
    #time.sleep(10)
    #break
    
#driver2.close()