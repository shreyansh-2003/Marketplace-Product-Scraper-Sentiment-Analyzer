# Importing the modules, methods created by me for
from webscrapping import *
from ProjectVisualisations import *
from ProductReviewAnalysis import *
import matplotlib.pyplot

matplotlib.pyplot.switch_backend('Agg')

from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__, template_folder='./templates', static_folder='./templates/static')


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/search_bar', methods=["POST", "GET"])
def search_bar():
    global product_name, brand_name
    if request.method == "POST":
        product_name = request.form.get("prod_name")
        brand_name = request.form.get("brand_name")
        print(product_name)
        print(brand_name)

    else:
        pass
    return redirect(url_for('loading', name=product_name))


@app.route('/loading+<name>')
def loading(name):
    return render_template('loading.html')


@app.route('/webScrape')
def webScrape():
    # execCode() from websrapping.py
    global df, brand_present_df
    df, brand_present_df = execCode(product_name, brand_name)
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df['index'] = df.index
    df.to_csv("sampled.csv")
    visualiseQueryReults(df, brand_present_df)
    return "Web Scrapping and Analysis Done !"


@app.route('/displayGraphs', methods=['GET', 'POST'])
def displayGraphs():
    # Product Count related functions
    amazon_prod_count = sum(df["ecommerce_website"] == "Amazon")
    flipkart_prod_count = sum(df["ecommerce_website"] == "Flipkart")
    actual_amazon_prod_count = sum(brand_present_df["ecommerce_website"] == "Amazon")
    actual_flipkart_prod_count = sum(brand_present_df["ecommerce_website"] == "Flipkart")

    g1m1 = "-> Amazon has " + str(amazon_prod_count) + " products listed on each page."
    g1m2 = "-> Flipkart has " + str(flipkart_prod_count) + " products listed on each page."

    g2m1 = "-> Amazon has " + str(
        amazon_prod_count) + " products listed on each page for the queried product,out of which " + str(
        amazon_prod_count - actual_amazon_prod_count) + " are of other brands"

    g2m2 = "-> Flipkart has" + str(flipkart_prod_count) + "products listed on each page for the queried product,out of " \
                                                          "which " + str(flipkart_prod_count -
                                                                         actual_flipkart_prod_count) + " are of other " \
                                                                                                       "brands "

    # Price Related
    amz_avg_price = subset_avg_gen(df, "Amazon", "product_price")
    flk_avg_price = subset_avg_gen(df, "Flipkart", "product_price")
    g3m1 = "-> Average Product Price for queried product at amazon is : " + str(round(amz_avg_price, 3))
    g3m2 = "-> Average Product Price for queried product at flipkart is : " + str(round(flk_avg_price, 3))

    # Rating Related
    amz_avg_ratings = subset_avg_gen(df, "Amazon", "product_rating")
    flk_avg_ratings = subset_avg_gen(df, "Flipkart", "product_rating")

    boxratm1 = "-> Average Product Rating for queried product at amazon is : " + str(round(amz_avg_ratings, 3))
    boxratm2 = "-> Average Product Rating for queried product at flipkart is : " + str(round(flk_avg_ratings, 3))

    # Correlation related
    price_rating_corr = df.corr()['product_rating']["product_price"]

    if abs(price_rating_corr) > 0.9:
        corr_msg = "-> Product Ratings and Product Prices are very hignly correlated. "

    if abs(price_rating_corr) > 0.7:
        corr_msg = "-> Product Ratings and Product Prices are hignly correlated. "

    if abs(price_rating_corr) > 0.5:
        corr_msg = "-> Product Ratings and Product Prices are moderately correlated. "

    if abs(price_rating_corr) > 0.3:
        corr_msg = "-> Product Ratings and Product Prices have low correlation. "

    else:
        corr_msg = "-> Product Ratings and Product Prices have very little to no correlation. "

    # Rendering Webpage
    return render_template('base_analysis.html', prd=product_name, brd=brand_name,
                           g1m1=g1m1,
                           g1m2=g1m2,
                           g2m1=g2m1,
                           g2m2=g2m2,
                           g3m1=g3m1,
                           g3m2=g3m2,
                           boxratm1=boxratm1,
                           boxratm2=boxratm2,
                           corr_msg=corr_msg
                           )


@app.route('/productsDisplay', methods=['GET', 'POST'])
def productsDisplay():
    arr_df = df.values.tolist()
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
    product_name = df.iloc[int(id)][2]

    df_cmts = mainReviiewsAnalysis(df[df["ecommerce_website"] == "Amazon"], product_name, comment_count)
    df_cmts['review_date'] = pd.to_datetime(df_cmts['review_date'])
    df_cmts.to_csv("cmts_sample.csv")
    visualiseCommentsAnalysis(df_cmts)

    return "Comments Scrapping  and Analysis Done !"


@app.route('/displaySentiGraphs')
def displaySentiGraphs():
    #pos_count = sum(df_cmts["sentiment"] == "Positive")
    #neu_count = sum(df_cmts["sentiment"] == "Neutral")
    #neg_count = sum(df_cmts["sentiment"] == "Negative")


   # pos = " -> Positive Sentiment Reviews : " + str(pos_count)
    #neg = " -> Negative Sentiment Reviews : " + str(neg_count)
    #neu = " -> Neutral Sentiment Reviews : " + str(neu_count)

    return render_template("reviews_analysis.html",
                           prod_name=df.iloc[id][2],
                           prod_price=df.iloc[id][3],
                           prod_rat=df.iloc[id][4],
                           prod_rat_count=df.iloc[id][5],
                           prod_url=df.iloc[id][7],
                           prod_img=df.iloc[id][6]
                           )


if __name__ == '__main__':
    app.run(use_reloader=False)