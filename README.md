# vectorai

A classification task for vector.ai

For this project, the task given was to a) scrape website resources to collect a dataset of vessel names, port locations, and company/corporations names, then generate a dataset of dates of different formats, then b) build and train a classifier model that can classify input text into one of these 4 categories (vessel, port, company, date). For the classifier architecture, a resource was provided for some neural networks written in TensorFlow.

For the data generation part of this exercise, I used the following methods. The python scripts for this section is found in the folder “Data Generation”.
Ports: I accessed this website directories.lloydslist.com, which contained a map with markers for each port worldwide. Searching the html source, I located the .json file that produced these markers at directories.lloydslist.com/script/stockist_map_details.json, and used python json and urllib to make a list of the “title” property of each object. I also built a scraper for the Wikipedia table of sovereign states, as the examples given in the introductory document included these as well.
Vessels: I used this website www.fleetmon.com/vessels and scraped the names column. This was done with a selenium Chrome web driver, and the names were located by their generic XPaths, which was found simply in the .html source. A challenge for this was that clicking ‘next’ on the list page did not change the url, as the page operated on Javascript, nor did the next button have a href url link. This meant I had to simulate mouse movements and clicks using selenium using:

    elem = driver.find_element_by_id(iteratorID)
    ActionChains(driver).move_to_element(elem).click(elem).perform()

where iteratorID was the argument for the next button’s ID. Next, I had to make sure that the next page of entries had loaded before scraping the page, as otherwise there would be a fatal error to the script. This again could not be done the standard way, so I had the driver check if the loading element (an animated navigator icon) had appeared and disappeared from the screen before scraping the page.

    WebDriverWait(driver, TIMEOUT).until(EC.presence_of_element_located((By.XPATH, LOADING_ELEMENT_XPATH)))
    WebDriverWait(driver, TIMEOUT).until_not(EC.presence_of_element_located((By.XPATH, LOADING_ELEMENT_XPATH)))

Corporations: For this one I used the given opencorporates.com resource. However, the front page requires a search query that only returns entries with an exact word match. This meant I could not iterate over each letter, as searching (say) “A” would only return companies with A as a word in their name. Instead, I went to the webpage they had for each country’s register and iterated over the pages for each list, then returned to the registers main page. While iterating over a countries list, the page would block after three ‘next’ clicks, so I reverted to the mage registers list after this happened.
Dates: This one was a fairly simple task of random generation, and any troubles I had with this one are illustrated and annotated in date_creator_old.py and date_creator.py

Next, I accessed GitHub user brightmart’s repo on various text classification neural network architectures written on TensorFlow. The readme on that repo suggested that the CNN was the most effective. This applies most to our task, as the text entries are short, and the RNNs’ short term memory capabilities are not needed as much. More important is the NN’s ability to find spatial patterns in the entries, which can make them distinct (easiest to imagine is the dates entries for their use of numbers).
Some changes to architecture had to be made also. In his CNN, brightmart uses tf.embedding_lookup to embed words in his text. I opted to embed the text before entering it into the neural network’s first layer as a batch of matrices. Given that we are working with short entries, and formatted dates, I split the entries into lists of characters, and embedded the characters using a word2vec model. To make all the entries the same size, I augmented the data before entering into the first layer. If the entry was longer than 40 characters, I would randomly average two adjacent character vectors until the word matrix was of size (40, 100), 100 being the embed size; if the matrix was too small, I would toss a coin to either randomly repeat short substrings in the entry, or find the closest letter vector to a random letter, and insert that into the matrix. The coin toss would be iterated until the right size was reached.
Some debugging also had to be done on the architecture layer sizes. While the code was there for all 4 combinations, his code was primed for a single layer CNN of multi-labelled data, wheras I wanted a double layered CNN for single labelled data. I may change this to single layer for memory costs later.

Things that could be added/improved/changed:
•	Including city names from Wikipedia for the place name database.
•	Find a way to override the timeout on the vessels Javascript list
•	Find a way to continue past 3 pages on the companies register for each country
•	Improve augmentation techniques for embedded strings
•	Change architecture to single layer (?)
