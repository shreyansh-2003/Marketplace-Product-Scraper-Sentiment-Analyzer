# Importing Required libraries and modules

from bs4 import BeautifulSoup
import requests
import time
import datetime

import smtplib
import re

import pandas as pd
import seaborn as sns
import datetime as dt
import numpy as np
# Finding user agent at: https://httpbin.org/get

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from IPython.display import clear_output
from tabulate import tabulate
import urllib.request
# from PIL import Image
from IPython.display import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)

# Change some of seaborn's style settings with `sns.set()`
sns.set(style="ticks",  # The 'ticks' style
        rc={"figure.figsize": (6, 9),  # width = 6, height = 9
            "figure.facecolor": "ivory",  # Figure colour
            "axes.facecolor": "ivory"})  # Axes colour

# Importing files created by me, and saved in. webscrapping.py
from ecom_prod_scraper import AmazonScraper
from ecom_prod_scraper import FlipkartScraper
from amz_comments_scraper import ScrapeComments

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


"""
User Defined Functions
"""


# User Defined Function To neatly print the products of the particular dataset
def printTableProducts(df, length):
    products_dict = dict(df[df["ecommerce_website"] == "Amazon"]["product_name"])
    arr = []

    for x in products_dict:
        arr.append([x, products_dict.get(x)[:70]])

        if len(arr) == length:
            break

    print(tabulate(arr, headers=['Product No.', 'Product Name']))


# User Defined Function To Show the selected products details in a neat manner
def showProductDeets(df, index):
    element = df.iloc[index]

    print((color.BOLD + color.BLUE + 'PRODUCT DETAILS' + color.END).center(100, "-"))
    print()
    print(color.BOLD + 'Product Name : ' + color.END, element[2])
    print(color.BOLD + 'Product Price : ' + color.END, element[3])
    print(color.BOLD + 'Average Product Rating : ' + color.END, element[4])
    print(color.BOLD + 'Total Number of Ratings : ' + color.END, element[5])
    print()

    urllib.request.urlretrieve(df.iloc[index][6], "ex.png")
    display(Image(filename='ex.png'))
    print("-" * 100)





# Function to add a new column that contains the size of each review
def addReviewLen(df):
    len_review = []
    for x in range(len(df)):
        rev_size = len(df.iloc[x][4])
        len_review.append(rev_size)
    df['review_size'] = len_review


# Function to clean the given comments dataset (drop na/null values)
def cleanCommentsDf(df):
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)


# Return Mapped value based on Categorical variables for weighted calculations
def mapValue(element):
    # 'Positive', 'Neutral', 'Negative' unqiue values
    if element == 'Positive':
        return 1

    if element == 'Neutral':
        return 0

    else:
        return -1


"""
VADER AND ROBERTA MODEL related user defined functions
"""


# Vader model (applying vader model on each comment and creating a df)
def getVaderDf(df):
    global sia
    sia = SentimentIntensityAnalyzer()
    # Runn the polarity score on the entire dataset
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Vader Model"):
        text = row['review_content']
        index = list(df.index)[i]
        val = sia.polarity_scores(text)
        res[index] = {
            'vader_neg': float(val['neg']),
            'vader_pos': float(val['pos']),
            'vader_comp': float(val['compound'])
        }

    # Converting thed dictionary into a dataframe and using transpose to make columns as rows
    vaders = pd.DataFrame(res).T
    vaders.reset_index(inplace=True, drop=True)
    # Merfing the vaders dataframe with the original dataset

    return vaders


# Roberta Model Scores (individual)
def polarity_scores_roberta(example):
    if len(example)>2150:
        example = example[:2000]
    #if len(example.split()) > 500:
        #example = " ".join(example.split()[:500])

    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# Roberta model (applying Roberta model on each comment and creating a df)
def getRobetaDf(df):

    # Runn the polarity score on the entire dataset
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Roberta Model"):
        text = row['review_content']
        index = list(df.index)[i]
        val = polarity_scores_roberta(text)
        res[index] = val

    # Converting thed dictionary into a dataframe and using transpose to make columns as rows
    df = pd.DataFrame(res).T
    df.reset_index(inplace=True, drop=True)

    return df


# User defined function to create a categorical column on the basis of the three Vader values
def getVaderSentiment(df):
    arr = []
    for x in range(len(df)):
        vader_compound_score = df["vader_comp"][x]

        if vader_compound_score >= 0.05:
            arr.append("Positive")

        elif vader_compound_score <= - 0.05:
            arr.append("Negative")

        else:
            arr.append("Neutral")

    if 'vader_sentiment' not in df.columns:
        df.insert(loc=6,
                  column='vader_sentiment',
                  value=arr)

    else:
        df['vader_sentiment'] = arr


# User defined function to create a categorical column on the basis of the three Roberta models
def getRobertaSentiment(df):
    arr = []
    for pos in range(len(df)):
        roberta_sentiment_dict = {"Positive": df[["roberta_pos", "roberta_neg", "roberta_neu"]].iloc[pos][0],
                                  "Negative": df[["roberta_pos", "roberta_neg", "roberta_neu"]].iloc[pos][1],
                                  "Neutral": df[["roberta_pos", "roberta_neg", "roberta_neu"]].iloc[pos][2]}

        roberta_sentiment_value = max(roberta_sentiment_dict, key=roberta_sentiment_dict.get)

        arr.append(roberta_sentiment_value)

    if 'roberta_sentiment' not in df.columns:
        df.insert(loc=6,
                  column='roberta_sentiment',
                  value=arr)

    else:
        df['roberta_sentiment'] = arr


# User defined function : Creating a weighted column based on
def getFinalSentiment(df):
    arr = []
    roberta_sentiment_vals = list(map(mapValue, list(df['roberta_sentiment'])))
    vader_sentiment_vals = list(map(mapValue, list(df['vader_sentiment'])))

    roberta_val = list(df['roberta_sentiment'])
    vader_val = list(df['vader_sentiment'])

    for x in range(len(df)):
        val = float((roberta_sentiment_vals[x] * 0.6) + (vader_sentiment_vals[x] * 0.4))

        if val >= 0.19:
            arr.append("Positive")

        elif val >= (-0.4):
            arr.append("Neutral")


        elif val >= (-1):
            arr.append("Negative")

    if 'sentiment' not in df.columns:
        df.insert(loc=6,
                  column='sentiment',
                  value=arr)

    else:
        df['sentiment'] = arr


"""
Plotting related user defined functions
"""


# Function for plotting a comparison on the kinds of reviews given for the product
def ratingsItemPlot(df):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)

    rev_count = list(df['review_ratings'].value_counts().sort_index())
    rev_tags = list(df['review_ratings'].unique())
    rev_tags.sort()

    plt.bar(rev_tags, rev_count)

    plt.plot(rev_tags, rev_count, color='black', linewidth=7.0)

    plt.xlabel('Review Stars', fontsize=16, color="red")
    plt.ylabel('Review Count', fontsize=16, color="red")

    plt.suptitle("Reviews Count for the product", fontsize=30, color='b')

    plt.subplot(1, 2, 2)
    #myexplode = [0.15, 0.19, 0.22, 0.12, 0.15]

    plt.pie(x=rev_count,
            shadow=True, autopct='%.0f%%')

    plt.legend(rev_tags)
    # plt.rcParams.update({'font.size': 10})

    plt.tight_layout()

    plt.savefig("templates/static/images/senti_5_ratingsPlot.png")

    return plt.show()


# User defined function to count and show a woordle like frequency chart
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.savefig("templates/static/images/senti_1_wordcloud.png")
    return plt.show()


# Graph that compares the review length and the ratings given
def sizeComPlot(df):
    plt.figure(figsize=(12, 7))
    plt.suptitle("Product Rating vs Review Length", fontsize=40, color="blue")
    plt.subplot(1, 2, 1)
    sns.pointplot(data=df, x="review_ratings", y="review_size")
    plt.xlabel("Rating", fontsize=16, color="red")
    plt.ylabel("Review Length", fontsize=16, color="red")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="review_ratings", y="review_size")
    plt.xlabel("Rating", fontsize=16, color="red")
    plt.ylabel("Review Length", fontsize=16, color="red")
    plt.tight_layout()

    plt.savefig("templates/static/images/senti_6_sizeCompPlot.png")

    return plt.show()


# User defined function to compare Vader and Roberta Model through dist plot
def compModelsGraph(df):
    plt.figure(figsize=(15, 8))
    plt.suptitle("In-Depth VADER and ROBERTA Model Analysis", color="blue", fontsize=30)

    plt.subplot(1, 3, 1)
    plot_one = sns.distplot(list(df['vader_pos']), kde_kws=dict(linewidth=7))
    plot_two = sns.distplot(df['roberta_pos'], kde_kws=dict(linewidth=7))
    plt.xlabel("POSITIVE SCORES", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    plt.subplot(1, 3, 2)
    plot_one = sns.distplot(list(df['vader_neg']), kde_kws=dict(linewidth=7))
    plot_two = sns.distplot(df['roberta_neg'], kde_kws=dict(linewidth=7))
    plt.xlabel("NEGATIVE SCORES", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    plt.subplot(1, 3, 3)
    plot_one = sns.distplot(list(df['vader_comp']), kde_kws=dict(linewidth=7))
    plot_two = sns.distplot(df['roberta_neu'], kde_kws=dict(linewidth=7))
    plt.xlabel("COMPOUND SCORES", color='red', fontsize=16)
    plt.ylabel("DENSITY", color='red', fontsize=16)
    plt.legend(["VADER MODEL", "ROBERTA MODEL"])

    plt.savefig("templates/static/images/senti_7_compModels.png")

    return plt.show()


# Plotting Vader Scores vs Ratings
def plotVaderResults(df):
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    sns.barplot(data=df, x='review_ratings', y='vader_comp', ax=axs[0], ci=None)
    axs[0].set_title("Compound Score (VADER)")

    sns.barplot(data=df, x='review_ratings', y='vader_neg', ax=axs[1], ci=None)
    axs[1].set_title("Negative Score (VADER)")

    sns.barplot(data=df, x='review_ratings', y='vader_pos', ax=axs[2], ci=None)
    axs[2].set_title("Positive Score (VADER)")

    plt.tight_layout()

    plt.savefig("templates/static/images/senti_8_plotVader.png")

    return plt.show()


# Plotting Roberta Model Scores vs Ratings
def plotRobertaResults(df):
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    sns.barplot(data=df, x=df['review_ratings'], y=df['roberta_neu'], ax=axs[0], ci=None)
    axs[0].set_title("Neutral Score (ROBERTA MODEL)")

    sns.barplot(data=df, x=df['review_ratings'], y=df['roberta_neg'], ax=axs[1], ci=None)
    axs[1].set_title("Negative Score (ROBERTA MODEL)")

    sns.barplot(data=df, x=df['review_ratings'], y=df['roberta_pos'], ax=axs[2], ci=None)
    axs[2].set_title("Positive Score (ROBERTA MODEL)")

    plt.tight_layout()

    plt.savefig("templates/static/images/senti_9_plotRoberta.png")

    return plt.show()


# Density Plot for final sentiment density (Date wise)
def sentimentDensityPlot(df):
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='review_date', hue="sentiment", palette="icefire",
                multiple="fill")
    plt.title("Weighted Sentiment Spread", fontsize=30, color='b')
    plt.xlabel("Review Date", fontsize=16, color="red")
    plt.ylabel("Density", fontsize=16, color="red")

    plt.savefig("templates/static/images/senti_3_densityPlot.png")

    return plt.show()


# Bar Chart to compare the Sentiment values of ROBERTA and VADER model
def modelValsComp(df):
    rob_vad_vals = list(df["vader_sentiment"]) + list(df["roberta_sentiment"])
    which_vals = ["VADER" for x in range(len(df))] + ["ROBERTA" for x in range(len(df))]
    temp_df = pd.DataFrame({"Sentiment": rob_vad_vals, "Model": which_vals})

    plt.figure(figsize=(12, 7))
    sns.histplot(data=temp_df, x="Sentiment", hue="Model", multiple="dodge", shrink=.8)
    plt.title("ROBERTA VS VADER MODEL SENTIMENT COMPARISON", fontsize=30, color='b')
    plt.xlabel("Sentiment", fontsize=16, color="red")
    plt.ylabel("Review Count", fontsize=16, color="red")
    plt.savefig("templates/static/images/senti_4_modelVals.png")
    return plt.show()


# Bar Chart and Pie Chart of weighted Sentiment Values
def sentimentPlot(df):
    pos_count = sum(df["sentiment"] == "Positive")
    neu_count = sum(df["sentiment"] == "Neutral")
    neg_count = sum(df["sentiment"] == "Negative")

    print("-> Positive Senitment Reviews : ", pos_count)
    print("-> Neutral Senitment Reviews : ", neu_count)
    print("-> Neutral Senitment Reviews : ", neg_count)

    plt.figure(figsize=(12, 7))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    plt.suptitle(" Weighted 3:2 (ROBERTA MODEL : VADER MODEL) COMP", fontsize=30, color='b')
    plt.subplot(1, 2, 1)

    plt.bar(x=["Positive", "Neutral", "Negative"],
            height=[pos_count, neu_count, neg_count],
            color=["green", 'yellow', 'red'])

    plt.xlabel("Sentiment", fontsize=16, color="red")
    plt.ylabel("Count of reviews", fontsize=16, color="red")

    plt.subplot(1, 2, 2)
    plt.pie(x=[pos_count, neu_count, neg_count], labels=["Positive", "Neutral", "Negative"],
            colors=['green', 'yellow', 'red'], autopct='%.0f%%')

    plt.tight_layout()
    plt.savefig("templates/static/images/senti_2_sentimentPlot.png")
    return plt.show()


"""
MAIN FUNCTION
"""


def mainReviiewsAnalysis(df, product_name, cmts_size):
    # Creating a subset of only Amazon related products
    amz_df = df[df["ecommerce_website"] == "Amazon"]

    # Scrapping the amazon product's review using the module created by me in Webscrapping.py
    comments_scrapping = ScrapeComments(amz_df, product_name, cmts_size)
    df = comments_scrapping.getDf()

    # Checking once more for null/na/redundant values in the freshly scraped df
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

    cleanCommentsDf(df)
    addReviewLen(df)
    # Using the user defined function to generate vader and roberta model values for the comments
    vader_df = getVaderDf(df)
    roberta_df = getRobetaDf(df)

    # Joining the initial data
    x = roberta_df.join(vader_df)
    df_cmt = df.join(x)
    getVaderSentiment(df_cmt)
    getRobertaSentiment(df_cmt)
    getFinalSentiment(df_cmt)

    return df_cmt


# Visualising Plots function with the help of the otehr methods created
def visualiseCommentsAnalysis(df_cmts):
    plt0 = show_wordcloud(list(df_cmts['review_content']))
    plt1 = sentimentPlot(df_cmts)
    plt2 = sentimentDensityPlot(df_cmts)
    plt3 = modelValsComp(df_cmts)
    plt4 = ratingsItemPlot(df_cmts)
    plt5 = sizeComPlot(df_cmts)
    plt6 = compModelsGraph(df_cmts)
    plt7 = plotVaderResults(df_cmts)
    plt8 = plotRobertaResults(df_cmts)
