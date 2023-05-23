# Importing required modules/libraries
import re
import time
import warnings
import pandas as pd
import regex as re
import requests
import seaborn as sns
from bs4 import BeautifulSoup
import concurrent.futures
warnings.filterwarnings("ignore")

# Find user agent at: https://httpbin.org/get

# amazon orange, flipkart yellow
colors = ["#FEBD69", "#F8E831"]
customPalette = sns.set_palette(sns.color_palette(colors))

# Change some of seaborn's style settings with `sns.set()`
sns.set(style="ticks",  # The 'ticks' style
        rc={"figure.figsize": (6, 9),  # width = 6, height = 9
            "figure.facecolor": "ivory",  # Figure colour
            "axes.facecolor": "ivory"},
        palette=customPalette)  # Axes colour

# dark black-blue amazon combo, flipkart blue
colors_2 = ["#37475A", "#047BD5"]

# Finding user agent at: https://httpbin.org/get

"""
Amazon Web Scraping Class
"""

class AmazonScraper:

    def __init__(self, productName, pg_no, headers):

        self.user_query = productName  # Storing the searched query as an attribute

        # Device Specific Headers
        self.head = headers

        # Driver Code as attributes (mandatory)
        querry = self.generateUrl(productName, pg_no)  # Getting the required querry
        soup_scrapped = self.connectUrl(querry)  # Connect to the amazon page and scrape it through BeautifulSoup
        # This will hold all the values of all
        self.cells = soup_scrapped.find_all('div', {'class': 's-result-item',
                                                    'data-component-type': 's-search-result'})

    # Creating a url request based on amazon's website url mechanisms
    def generateUrl(self, query, pg_no):
        query = re.sub(r"\s+", '+', query)
        amazon_url = 'https://www.amazon.in/s?k={0}&page={1}'.format(query, pg_no)
        return amazon_url

    # connecting to the website
    def connectUrl(self, url):
        # s= requests.Session()
        page = requests.get(url, headers=self.head)
        soup = BeautifulSoup(page.content, "html.parser")
        return soup

    # Function to check whether no. of instances haven't exceeded 20
    def lenCheck(self, arr):
        return False

    #         if len(arr)==20:
    #             return
    #         else:
    #             return False

    # Getter method to retrieve title(name) of each listed product
    def getTitles(self):
        product_titles = []

        for results in self.cells:
            item_name = results.h2.text  # product title (static html --> h2)
            product_titles.append(item_name)

            if self.lenCheck(product_titles):
                break

        return product_titles

        # Getter method to retrieve price(in rs.) of each listed product

    def getPrices(self):
        product_prices = []

        for results in self.cells:
            try:
                item_price = results.find('span', {'class': "a-price-whole"}).text  # (static html --> span tag)
                item_price = float(item_price.replace(",", ""))


            except AttributeError:
                item_price = None

            product_prices.append(item_price)

            if self.lenCheck(product_prices):
                break

        return product_prices

    # Getter method to retrieve review ratings of each listed product
    def getRatings(self):
        product_ratings = []

        for results in self.cells:
            try:
                item_rating = results.find('i', {'class': 'a-icon'}).text

                # Exception, if there are no reviews or ratings avaliable
                try:
                    item_rating = float(item_rating[:3])

                except ValueError:
                    item_rating = 0

            except AttributeError:
                item_rating = 0

            product_ratings.append(item_rating)

            if self.lenCheck(product_ratings):
                break

        return product_ratings

    # Getter method to retrieve number of reviews of each listed product
    def getRatingsCount(self):
        product_ratings_count = []

        for results in self.cells:
            try:
                item_rating_count = results.find('span', {'class': "a-size-base s-underline-text"}).text

                # Exception, if there are no reviews or ratings available
                try:
                    item_rating_count = int(item_rating_count.replace(",", ""))

                except ValueError:
                    item_rating_count = 0

            except AttributeError:
                item_rating_count = 0

            product_ratings_count.append(item_rating_count)

            if self.lenCheck(product_ratings_count):
                break

        return product_ratings_count

    # Getter method to retrieve images of each listed product
    def getImages(self):
        product_images = []
        for results in self.cells:
            try:
                item_image = results.a.img["src"]

            except TypeError:
                item_image = None

            except AttributeError:
                item_image = None

            product_images.append(item_image)

            if self.lenCheck(product_images):
                break

        return product_images

    # Getter method to retrieve images of each listed product
    def getUrls(self):
        product_urls = []

        for results in self.cells:
            try:
                item_url = "https://www.amazon.in" + results.h2.a["href"]

            except AttributeError:
                item_url = None

            product_urls.append(item_url)

            if self.lenCheck(product_urls):
                break

        return product_urls

    # Converting the scrapped data into an appropriate dataframe
    def toDataframe(self):

        # Printing the array lengths while testing for discrapancies
        """
        print(len(self.getTitles()))
        print(len(self.getUrls()))
        print(len(self.getPrices()))
        print(len(self.getRatings()))
        print(len(self.getRatingsCount()))
        print(len(self.getImages()))
        """

        df = pd.DataFrame({
            "querry_searched": self.user_query,
            "ecommerce_website": "Amazon",
            "product_name": self.getTitles(),
            "product_price": self.getPrices(),
            "product_rating": self.getRatings(),
            "rating_count": self.getRatingsCount(),
            "poduct_image_url": self.getImages(),
            "product_url": self.getUrls()
        })

        return df

    # Additional function to go to each product's url and extract brand name, request response time too long
    def addBrand(self, df):
        product_brands = []

        for x in range(len(df)):
            url = df.loc[x][7]
            page = requests.get(url, headers=self.head)
            soup = BeautifulSoup(page.content, "html.parser")

            try:
                product_overview = soup.find(id="productOverview_feature_div")
                cell_detailed = product_overview.find('div', {"class": "a-section a-spacing-small a-spacing-top-small"})
                cells = cell_detailed.find_all('span')

                y = "Brand"
                for x in range(0, len(cells)):
                    if y in cells[x]:
                        item_brand = cells[x + 1].text


            except AttributeError:
                item_brand = 0

            product_brands.append(item_brand)

        df["product_brand"] = product_brands


"""
Flipkart Web Scraping Class
"""


class FlipkartScraper:

    def __init__(self, productName, pg_no, headers):

        self.grid = False
        self.img = True

        self.user_query = productName  # Storing the searched querry as an attribute

        # Device Specific Headers
        self.head = headers

        # Driver Code as attributes (mandatory)
        querry = self.generateUrl(productName, pg_no)  # Getting the required querry
        self.soup_scrapped = self.connectUrl(querry, headers)  # Connect to the Flipkart page and scrape it through BeautifulSoup
        # This will hold all the values of all cells in the page
        self.cells = self.soup_scrapped.find_all('a', {"class": "_1fQZEK"})

        # Flagging self.grid incase the layout of flipkart webpage is gridwise, instead of row wise
        if len(self.cells) == 0:
            self.grid = True
            self.cells = self.soup_scrapped.find_all('div', {"class": "_4ddWXP"})

            if len(self.cells) == 0:
                self.grid = True
                self.img = False
                self.cells = self.soup_scrapped.find_all('div', {"class": "_1xHGtK _373qXS"})


        else:
            self.grid = False

    # Creating a url request based on amazon's website url mechanisms
    def generateUrl(self, query, pg_no):
        query = re.sub(r"\s+", '+', query)
        flipkart_url = 'https://www.flipkart.com/search?q={0}&page={1}'.format(query, pg_no)
        return flipkart_url

    # connecting to the website
    def connectUrl(self, url, headers):
        page = requests.get(url, headers=self.head)
        soup = BeautifulSoup(page.content, "html.parser")
        return soup

    # Getter method to retrieve title(name) of each listed product
    def getTitles(self):
        product_titles = []

        for index in range(len(self.cells)):
            try:
                if self.grid:
                    item_title = self.cells[index].find_all('a')[1]['title']

                else:
                    item_title = self.cells[index].find('div', {"class": "_4rR01T"}).text

            except AttributeError:
                item_title = 0

            except KeyError:
                item_title = 0

            except ValueError:
                item_title = 0

            product_titles.append(item_title)

        return product_titles

    # Getter method to retrieve price(in rs.) of each listed product
    def getPrices(self):
        product_prices = []
        for index in range(len(self.cells)):
            try:
                if self.grid:
                    str_price = self.cells[index].find_all('a')[2].find('div', {"class": "_30jeq3"}).text
                    item_price = float(str_price[1:].replace(",", ""))

                else:
                    item_price = (self.cells[index].find('div', {"class": "_30jeq3 _1_WHN1"}).text)
                    item_price = float((item_price[1:]).replace(",", ""))

            except AttributeError:
                item_price = None

            product_prices.append(item_price)

        return product_prices

    # Getter method to retrieve review ratings of each listed product
    def getRatings(self):
        product_ratings = []
        for index in range(len(self.cells)):

            try:
                if self.grid:
                    item_rating = float(self.cells[index].find('div', {'class': '_3LWZlK'}).text)

                else:
                    item_rating = float(self.cells[index].find('div', {"class": "_3LWZlK"}).text)

            except AttributeError:
                item_rating = 0

            except ValueError:
                item_rating = 0

            product_ratings.append(item_rating)

        return product_ratings

    # Getter method to retrieve number of ratings of each listed product
    def getRatingsCount(self):
        product_ratings_count = []
        for index in range(len(self.cells)):

            try:

                if self.grid:
                    str_ratings_count = self.cells[index].find('span', {'class': '_2_R_DZ'}).text
                    item_rating_count = int(str_ratings_count.replace("(", "").replace(")", ""))

                else:
                    ratings_text = self.cells[index].find('span', {"class": "_2_R_DZ"}).text
                    split_position = re.search(" ", ratings_text).start()  # Returns position of first space
                    item_rating_count = int(ratings_text[:split_position].replace(",", ""))


            except ValueError:
                item_rating_count = 0

            except AttributeError:
                item_rating_count = 0

            product_ratings_count.append(item_rating_count)

        return product_ratings_count

    # Getter method to retrieve number of reviews/comments of each listed product
    def getReviewsCount(self):
        product_reviews_count = []
        for index in range(len(self.cells)):
            try:

                if self.grid:
                    item_reviews_count = None
                    # no. of reviews not available in main catalogue grid page

                else:
                    reviews_text = self.cells[index].find('span', {"class": "_2_R_DZ"}).text
                    split_position = re.search(r"\xa0\b", reviews_text).start()
                    item_reviews_count = reviews_text[
                                         split_position + 1:-8]  # we are slicing 8 characters from the end for " Reiews"



            except ValueError:
                item_reviews_count = 0

            except AttributeError:
                item_reviews_count = 0

            product_reviews_count.append(item_reviews_count)

        return product_reviews_count

    # Getter method to retrieve images of each listed product
    def getImages(self):
        product_image_urls = []
        for index in range(0, len(self.cells)):
            try:
                if self.img:
                    item_image_url = self.cells[index].find('img', {'class': '_396cs4'})['src']

                else:
                    item_image_url = self.cells[index].find('img', {'class': '_2r_T1I'})['src']


            except:
                item_image_url = None

            product_image_urls.append(item_image_url)

        return product_image_urls

    # Getter method to retrieve images of each listed product
    def getUrls(self):
        product_urls = []

        for index in range(len(self.cells)):
            try:

                if self.grid:
                    item_qrr = self.cells[index].find_all('a')[1]['href']
                    item_url = "https://www.flipkart.com" + item_qrr

                else:
                    item_url = self.cells[index]['href']

            except AttributeError:
                item_url = None

            except KeyError:
                item_url = None

            product_urls.append(item_url)

        return product_urls

    # Converting the scrapped data into an appropriate dataframe
    def toDataframe(self):

        # Printing the array lengths while testing for discrapancies
        """
        print(len(self.getTitles()))
        print(len(self.getUrls()))
        print(len(self.getPrices()))
        print(len(self.getRatings()))
        print(len(self.getRatingsCount()))
        print(len(self.getReviewsCount()))
        print(len(self.getImages()))
        """

        df = pd.DataFrame({
            "querry_searched": self.user_query,
            "ecommerce_website": "Flipkart",
            "product_name": self.getTitles(),
            "product_price": self.getPrices(),
            "product_rating": self.getRatings(),
            "rating_count": self.getRatingsCount(),
            "poduct_image_url": self.getImages(),
            "product_url": self.getUrls(),
            "reviews_count": self.getReviewsCount()
        })

        return df


"""
Functions for Creating and merging datasets acccording to a user's querry
"""


# Function that concats 'n' dataframes
def joinDf(df_arr):
    df = pd.DataFrame()
    for x in df_arr:
        df = pd.concat([df, x],
                       ignore_index=True)
    return df


# User Defined Funciton : that checks whether the brand is present in the prodcut_name
def brandCheck(df, brand):
    if len(brand) == 0:
        return df

    else:
        # Creating a regex expression so that if there's upper or lower case in brand or product it'll be accounted
        letters = "("
        for letter in brand:
            upper_x = letter.upper()
            lower_x = letter.lower()
            letters = letters + "[" + upper_x + lower_x + "]"

        letters = letters + "*)\s"
        # print(letters)

        arr_vals = []
        for x in range(0, len(df)):
            product_name = str(df.loc[x][2])
            index_brand = re.search(letters, product_name)
            if index_brand:
                arr_vals.append(df.loc[x])

        df = pd.DataFrame(arr_vals, columns=list(df.columns))

        return df


# Function that combines the amazon and filpkart webscrapped
def finalDf(amz_df, flk_df, brand_name):
    arr_dfs = []
    arr_dfs.append(amz_df)
    arr_dfs.append(flk_df)
    df1 = joinDf(arr_dfs)
    df = brandCheck(df1, brand_name)
    # df = removePercentile(df,20)
    df.reset_index(inplace=True, drop=True)
    # df.drop(['index'],axis=1,inplace=True)

    return df1, df


# MAIN FUNCTION that takes in the user query
def exec_code(query, brand, t, headers):
    # A variable that creates a specified second buffer between current time and the end.s
    t_end = time.time() + t
    # Initialize page number
    pg_no = list(range(1, 100))
    amz_df_arr = []
    flk_df_arr = []

    # Function to scrape data from Amazon and return a dataframe
    def scrape_amazon(page):
        amz = AmazonScraper(query, page, headers)
        return amz.toDataframe()

    # Function to scrape data from Flipkart and return a dataframe
    def scrape_flipkart(page):
        flk = FlipkartScraper(query, page, headers)
        return flk.toDataframe()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        

        while time.time() < t_end:
            try:
                # Submit tasks to the executor
                amz_futures = [executor.submit(scrape_amazon, page) for page in pg_no if time.time() < t_end]
                flk_futures = [executor.submit(scrape_flipkart, page) for page in pg_no if time.time() < t_end]

                # Get the results of the completed futures
                amz_results = [future.result() for future in concurrent.futures.as_completed(amz_futures) if time.time() < t_end]
                flk_results = [future.result() for future in concurrent.futures.as_completed(flk_futures) if time.time() < t_end]

                # Append the results to the respective arrays
                amz_df_arr.extend(amz_results)
                flk_df_arr.extend(flk_results)

                pg_no = [page + len(pg_no) for page in pg_no]

            except Exception as e:
                continue

    # Joining both the amazon and flipkart pages individually
    amz_df = joinDf(amz_df_arr)
    flk_df = joinDf(flk_df_arr)

    df, brand_present_df = finalDf(amz_df, flk_df, brand)

    return df, brand_present_df
