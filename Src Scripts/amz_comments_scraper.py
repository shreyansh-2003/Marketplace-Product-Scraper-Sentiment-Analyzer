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

# Finding user agent at: https://httpbin.org/get

"""
Comments Scraping (Extraction) (Currently Amazon Specific)
"""


class ScrapeComments:

    def __init__(self, cmts_url, max_reviews, headers):

        # Device Specific Headers
        self.head = headers

        self.df_comments = self.getAmazonComments(cmts_url, max_reviews)

    # Function that returns the Data Frame created (extracted comments)
    def getDf(self):
        return self.df_comments

    # Main Function that extracts and converts the comments into a structured dataframe
    def getAmazonComments(self, cmts_url, max_reviews):

        # Counters
        page_no = 0  # All reviews --> current page number counter
        reviews = 0  # Total reviews extracted counter
        reviews_count = 0  # Counter initialised to prevent item called before assignment error

        # Empty lists that will hold all reviewee data like name, date of review, ratings gien, review title, content
        customer_name = []
        review_date = []
        ratings = []
        review_title = []
        review_content = []
        # global reviews_soup

        # Finding the url to the queried product name using indexing
        #item_url = list(df['product_url'][df['product_name'] == onclick_name])[0]

        item_url = cmts_url

        # Connecting (sending a request) to the product's page
        item_page = requests.get(item_url, headers=self.head)
        item_soup = BeautifulSoup(item_page.content, "html.parser")

        try:

            # Till the reviews counter is less than the reviews_count mentioned on the 'all reviews' page
            while reviews <= reviews_count:

                # Finding the link to "all reviews on the product's page", and adding iterating page_no variable
                reviews_url = item_soup.find('a', {'class': 'a-link-emphasis a-text-bold'})['href']
                reviews_url = "https://www.amazon.in" + reviews_url + "&pageNumber=" + str(page_no)

                # Sending request for the Reviews Url page
                reviews_page = requests.get(reviews_url, headers=self.head)
                reviews_soup = BeautifulSoup(reviews_page.content, "html.parser")


                # Number of reviews for the particular page, same throughout all comment (all reviews) pages
                reviews_count = reviews_soup.find('div', {'class': "a-row a-spacing-base a-size-base"}).text
                reviews_count = int(
                    (reviews_count.strip().split('ratings,'))[1].split('with')[0].strip().replace(",", ""))
                print(reviews_count)
                # Extracting the required comments in a page
                # These are generally lists containing 10 review related data each.
                names = reviews_soup.select('span.a-profile-name')[2:]
                titles = reviews_soup.select('a.review-title span')
                dates = reviews_soup.select('span.review-date')[2:]
                stars = reviews_soup.select('i.review-rating span.a-icon-alt')[2:]
                content = reviews_soup.select('span.review-text-content span')

                # IndexError Handling : If the sections are left empty by the customers, the index is out of range
                for count in range(len(dates)):

                    try:
                        customer_name.append(names[count].get_text())
                    except IndexError:
                        customer_name.append(0)

                    try:
                        review_date.append(dates[count].get_text().replace("Reviewed in India on ", ""))
                    except IndexError:
                        review_date.append(0)

                    try:
                        review_title.append(titles[count].get_text())
                    except IndexError:
                        review_title.append(None)

                    try:
                        ratings.append(float(stars[count].get_text()[:2]))
                    except IndexError:
                        ratings.append(None)

                    try:
                        review_content.append(content[count].get_text())
                    except IndexError:
                        review_content.append(None)

                    reviews = reviews + 1

                # Incase the number of reviews counter has reached the maximum reviews to be extracted
                if reviews == max_reviews:
                    break

                page_no = page_no + 1
                # print(page_no)


        except:
            pass

        """
        Debugging Section
        print(len(customer_name))
        print(len(review_date))
        print(len(review_title))
        print(len(ratings))
        print(len(review_content))
        """

        # Recalling required functions from the module.
        df = self.commentsToDf(customer_name, review_date, review_title, ratings, review_content)
        df = self.cleanReviewDates(df)

        return df

    # Function that assigns the lists extracted to a dataframe and returns it.
    def commentsToDf(self, names, dates, titles, ratings, content):

        reviews_df = pd.DataFrame({"customer_name": names,
                                   "review_date": dates,
                                   "review_title": titles,
                                   "review_ratings": ratings,
                                   "review_content": content})

        return reviews_df

    # Function to convert dates which have indian flag and incorrect redundant formats in it, st object in dd/mm/yy
    def cleanReviewDates(self, df):

        cleaned_dates = []
        for comment in list(df['review_date']):
            for string in range(len(comment)):
                if comment[string].isnumeric():
                    cleaned_dates.append(comment[string:])
                    break

        df['review_date'] = cleaned_dates
        df['review_date'] = pd.to_datetime(df['review_date']).dt.strftime('%d/%m/%Y')

        return df
