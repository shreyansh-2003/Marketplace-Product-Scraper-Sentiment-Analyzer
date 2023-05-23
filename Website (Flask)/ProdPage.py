# Importing the modules, methods created by me for
# from webscrapping import AmazonScraper
# from webscrapping import FlipkartScraper
# from webscrapping import ScrapeComments
# from webscrapping import *
from ProjectVisualisations import *
from ProductReviewAnalysis import *
import json
import pandas as pd

import matplotlib.pyplot

matplotlib.pyplot.switch_backend('Agg')

from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__, template_folder='./templates', static_folder='./templates/static')
df = pd.read_csv("sampled.csv")
arr_df = df.values.tolist()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('products_display.html', df=arr_df)


@app.route('/sentimentAnalysis', methods=['POST'])
def sentimentAnalysis():
    global id
    id = int(request.form.get('name'))
    print(id)
    return ""


@app.route('/cmts_option')
def cmts_option():
    return render_template('cmts_option.html')


@app.route('/load_web_scrape', methods=['POST'])
def load_web_scrape():
    global comment_count
    comment_count = request.form.get("cmt_count")
    print(comment_count)

    return render_template('loading2.html')


@app.route('/commentsScrape')
def commentsScrape():
    # execCode() from websrapping.py
    global df_cmts
    product_name = df.iloc[int(id)][3]

    df_cmts = mainReviiewsAnalysis(df[df["ecommerce_website"] == "Amazon"], product_name, comment_count)
    df_cmts['review_date'] = pd.to_datetime(df_cmts['review_date'])

    visualiseCommentsAnalysis(df_cmts)

    return "Comments Scrapping  and Analysis Done !"


@app.route('/displaySentiGraphs')
def displaySentiGraphs():
    pos_count = sum(df_cmts["sentiment"] == "Positive")
    neu_count = sum(df_cmts["sentiment"] == "Neutral")
    neg_count = sum(df_cmts["sentiment"] == "Negative")

    prod_name = df.iloc[id][3]
    prod_price = df.iloc[id][4]
    prod_rat = df.iloc[id][5]
    prod_rat_count = df.iloc[id][6]
    prod_url = df.iloc[id][8]
    prod_img = df.iloc[id][7]

    pos = " -> Positive Sentiment Reviews : " + str(pos_count)
    neg = " -> Negative Sentiment Reviews : " + str(neg_count)
    neu = " -> Neutral Sentiment Reviews : " + str(neu_count)

    return render_template("reviews_analysis.html",
                           prod_name=df.iloc[id][3],
                           prod_price=df.iloc[id][4],
                           prod_rat=df.iloc[id][5],
                           prod_rat_count=df.iloc[id][6],
                           prod_url=df.iloc[id][8],
                           prod_img=df.iloc[id][7],
                           pos=pos,
                           neg=neg,
                           neu=neu
                           )


if __name__ == '__main__':
    app.run(use_reloader=False, port=8001)
