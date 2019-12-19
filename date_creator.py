# For this date generator I will only use the formats given in the document plus a few others,
# as there are so many more formats you could opt for
# For reference, these are: (DD/MM/YYYY, MM/DD/YYYY, ;;Month DDth/st/rd YYYY, ;;Month DD YYYY)
# And I'll add:        ;;Year, Month D      //DDMMYY     //MMDDYY      //YYMMDD  
# The ones with // before them mean they will only use numberMonth, not the months names
# The ones with ;; before them mena they will only use month names, not numberMonth

# For each month name, I will do full and short names, and for each day/month number I will do 1 and 2 digits (i.e. leading zero or not)
# These will not be explicitly made as new types, but the numbers will be included there to be chosen

import random

months = ['January', 'February', 'March', 'April', 'May', 'June',
 'July', 'August', 'September', 'October', 'November', 'December',
 'Jan', 'Feb', 'Mar', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'] 
 
numberMonths = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

days = list(range(31)) + ['01', '02', '03', '04', '05', '06', '07', '08', '09']

years = list(range(1500, 2050))

lst = []
for i in range(1000):
    a = str(i)
    while len(a)<4:
        a = '0' + a
    lst.append(a)               # I will give this set of years its own list and give it a lower representation in generation, 
                                # as I don't want the RNN to be overly trained on short years or years starting with 0/00/000

oldyears = list(range(1500)) + lst

shortyears = [i[2:] for i in lst[0:100]]

# I will now generate 180 dates for each format using a full year, 120 more using short years (00, 99 etc.), 
# and 80 more using old years. With 5 general types, this will give a list of 1900 dates

# I will then repeat this process for numberMonths, giving a further 1900 for its 5 general types

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
    l = [(years, 180), (shortyears, 120), (oldyears, 80)]           # Since lists are unhashable for dictionary

    outlistfin = []
    for t in l:
        a = numberMonthDateGenerator(days, numberMonths, t[0], t[1])
        b = nameMonthDateGenerator(days, months, t[0], t[1])
        outlistfin += a
        outlistfin += b
    
    return outlistfin

print(len(generateDates()))