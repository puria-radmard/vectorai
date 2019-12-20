# After consulting Nisarg about year spread

import random

months = ['January', 'February', 'March', 'April', 'May', 'June',
 'July', 'August', 'September', 'October', 'November', 'December',
 'Jan', 'Feb', 'Mar', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'] 
 
numberMonths = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

days = list(range(31)) + ['01', '02', '03', '04', '05', '06', '07', '08', '09']

years = list(range(2000, 2050))

oldyears = list(range(1900, 2000))

# I will now generate 300 dates for each format using a full year, 100 more using short years (00, 99 etc.), 
# and 80 more using old years. With 5 general types, this will give a list of 2000 dates

# I will then repeat this process for numberMonths, giving a further 2000 for its 5 general types

def ddmmyySlash(d, m, y):
    return("{}/{}/{}".format(d, m, y))

def mmddyySlash(d, m, y):
    return("{}/{}/{}".format(d, m, y))

def verboseDate(d, m, y):
    if str(d)[-1] == 1: tag = "st"
    if str(d)[-1] == 2: tag = "nd"
    if str(d)[-1] == 3: tag = "rd"
    else: tag = "th"

    return("{} {}{} {}".format(m, d, tag, y))

def verboseDateShort(d, m, y):
    return("{} {} {}".format(m, d, y))

def inverseDate(d, m, y):
    return("{}, {} {}".format(y, m, d))

def ddmmyy(d, m, y):
    return("{}{}{}".format(d, m, y))

def mmddyy(d, m, y):
    return("{}{}{}".format(m, d, y))
    
def yymmdd(d, m, y):
    return("{}{}{}".format(y, m, d))


# Finally, we can group these functions into one for numberMonth, and ones for month names:

def nameMonthDateGenerator(days, months, years, number):
    funcs = (ddmmyySlash, mmddyySlash, verboseDate, verboseDateShort, inverseDate)
    outlist = []

    for f in funcs:
        for n in range(number):
            d = random.choice(days)
            m = random.choice(months)
            y = random.choice(years)
            
            outlist.append(f(d, m, y))
    
    return outlist

def numberMonthDateGenerator(days, months, years, number):
    funcs = (ddmmyySlash, mmddyySlash, ddmmyy, mmddyy, yymmdd)
    outlist = []

    for f in funcs:
        for n in range(number):
            d = random.choice(days)
            m = random.choice(months)
            y = random.choice(years)
            
            outlist.append(f(d, m, y))
    
    return outlist

# FINALLY finally, we can put this all together with the preoportions mentioned

def generateDates():

    
    a = numberMonthDateGenerator(days, numberMonths, years, 300)
    b = numberMonthDateGenerator(days, numberMonths, oldyears, 100)
    c = nameMonthDateGenerator(days, months, years, 300)
    d = nameMonthDateGenerator(days, months, oldyears, 100)

    outlistfin = a + b + c + d
    
    return outlistfin

l = generateDates()
d = {"data": l, "type": ["Date" for x in l]}

import pandas as pd

outcsv = pd.DataFrame(data = d)
outcsv.to_csv("C:/Users/puria/source/repos/puria-radmard/vectorai/Latest Data/dates.csv")