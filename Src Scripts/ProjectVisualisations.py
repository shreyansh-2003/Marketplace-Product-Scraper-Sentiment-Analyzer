# Importing required libraries for further analysis
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import random
import matplotlib.pyplot as plt
import warnings
import time
import plotly.express as px
import matplotlib.patheffects as pe
import regex as re

warnings.filterwarnings("ignore")

# Finding user agent at: https://httpbin.org/get

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


# Function to count products with brand name present in comparison to the original web - scrapped dataset
def countActualProductsPlot(df, df_correct):
    #     if len(df)==len(df_correct):
    #         myexplode=[0.5,0]
    #         plt.pie(x = [100,0], explode = myexplode, shadow = True,
    #             colors = ["g","r"])
    #         plt.legend(["Relevant Products","Irrelevant Products"])
    #         plt.title("No brand mentioned / All Products Have Brand Present in Name",
    #                   fontsize=30,color="b")
    #         return plt.show()
    #     else:
    try:
        amazon_prod_count = sum(df["ecommerce_website"] == "Amazon")
        flipkart_prod_count = sum(df["ecommerce_website"] == "Flipkart")

        actual_amazon_prod_count = sum(df_correct["ecommerce_website"] == "Amazon")
        actual_flipkart_prod_count = sum(df_correct["ecommerce_website"] == "Flipkart")

        print("-> Amazon has",
              amazon_prod_count,
              "products listed on each page for the queried product,out of which ",
              (amazon_prod_count - actual_amazon_prod_count),
              " are of other brands")

        print("-> Flipkart has",
              flipkart_prod_count,
              "products listed on each page for the queried product,out of which ",
              actual_flipkart_prod_count,
              " are of other brands")

        plt.figure(figsize=(12, 7))
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.50
        plt.suptitle("Stacked Bar Chart Comparing Relevance of Listed Products", fontsize=30, color='b')

        # Bar Plot
        plt.subplot(1, 2, 1)
        x = ["Amazon", "Flipkart"]
        y1 = [amazon_prod_count - actual_amazon_prod_count,
              flipkart_prod_count - actual_flipkart_prod_count]

        y2 = [actual_amazon_prod_count, actual_flipkart_prod_count]


        # plot bars in stack manner
        plt.bar(x, y1, color='red', edgecolor=colors_2)
        plt.bar(x, y2, bottom=y1, color=colors, edgecolor=colors_2)

        # Labels
        plt.xlabel("E-Commerce Website", fontsize=16, color="red")
        plt.ylabel("Count of Products", fontsize=16, color="red")
        plt.legend(["Irrelevant Products", "Relevant Products Amazon", "Relevant Products Flipkart"])

        # Pie Chart
        plt.subplot(1, 2, 2)
        myexplode = [0.4, 0]

        relv_prd = sum(y2) / (amazon_prod_count + flipkart_prod_count)
        irrelv_prd = 1 - relv_prd

        plt.pie(x=[relv_prd, irrelv_prd], explode=myexplode, shadow=True,
                colors=["g", "r"])
        plt.legend(["Relevant Products", "Irrelevant Products"])
        plt.tight_layout()


        return plt.show()

    except:
        print("Error !")


def countProductsPlot(df):
    amazon_prod_count = sum(df["ecommerce_website"]=="Amazon")
    flipkart_prod_count = sum(df["ecommerce_website"] == "Flipkart")

    print("-> Amazon has", amazon_prod_count, "products listed on each page.")
    print("-> Flipkart has", flipkart_prod_count, "products listed on each page.")

    plt.figure(figsize=(12, 7))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    plt.suptitle("Products Listed Per Page for the Query", fontsize=30, color='b')
    plt.subplot(1, 2, 1)
    # fig.subtitle('Products per page for given product')

    sns.countplot(x='ecommerce_website', data=df, color=customPalette,
                  linewidth=3, edgecolor=colors_2)

    plt.xlabel("Ecommerce Website", fontsize=16, color="red")
    plt.ylabel("No. of Products per page", fontsize=16, color="red")

    plt.subplot(1, 2, 2)
    plt.pie(x=[amazon_prod_count,flipkart_prod_count], labels=["Amazon", "Flipkart"],
            colors=customPalette, autopct='%.0f%%')

    plt.tight_layout()

    return plt.show()


# Function to find average of a clolumn for the particular e-commerce website
def subset_avg_gen(df, website, col):
    df_subset = df[df["ecommerce_website"] == website]
    average = df_subset[col].mean()
    return average


# User Defined Function : Average Price of the products displayed upon query, website wise
def priceCompLine(df):
    x = list(df[df["ecommerce_website"] == "Amazon"]["product_price"])
    y = list(df[df["ecommerce_website"] == "Flipkart"]["product_price"])
    plt.figure(figsize=(15, 10))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    overlapping = 0.1

    plt.plot(x, lw=6, path_effects=[pe.SimpleLineShadow(shadow_color=colors_2[0]), pe.Normal()])
    plt.plot(y, lw=6, path_effects=[pe.SimpleLineShadow(shadow_color=colors_2[1]), pe.Normal(), ])

    plt.legend(["Amazon", "Flipkart"], loc='upper left')
    plt.xlabel("Product No.", color="red")
    plt.ylabel("Price (in rs.)", color="red")
    plt.title("Comparing Prices (line-plot)", fontsize=30, c='b')


    return plt.show()


def CompPriceBox(df):
    amz_avg_price = subset_avg_gen(df, "Amazon", "product_price")
    flk_avg_price = subset_avg_gen(df, "Flipkart", "product_price")

    print("Average Product Price for queried product at amazon is : ", amz_avg_price)
    print("Average Product Price for queried product at flipkart is : ", flk_avg_price)

    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    plt.figure(figsize=(15, 10))
    # Box plot
    b = sns.boxplot(data=df,
                    x="ecommerce_website",
                    y="product_price",
                    width=0.4,
                    linewidth=2,

                    showfliers=False)
    # Strip plot

    b = sns.stripplot(data=df,
                      x="ecommerce_website",
                      y="product_price",
                      # Colours the dots
                      linewidth=1,
                      alpha=0.4)

    b.set_ylabel("Price", fontsize=14, color="red")
    b.set_xlabel("E-Commerce Website", fontsize=14, color="red")
    b.set_title("Comparing Prices (box-plot)", fontsize=30, c='b')

    sns.despine(offset=5, trim=True)

    return b.get_figure()


# User Defined Function : Number of products available on Amazon and flipkart for the given querry
def priceNamePlotly(df):
    fig = px.line(df, x="product_name", y="product_price",
                  color="ecommerce_website", template="plotly_dark",
                  color_discrete_sequence=colors,
                  title="Products and their price")

    fig.update_xaxes(showticklabels=False)


    return fig.show()


# User Defined Function : Average Rating of the products displayed upon query, website wise
def CompRatingBox(df):
    amz_avg_ratings = subset_avg_gen(df, "Amazon", "product_rating")
    flk_avg_ratings = subset_avg_gen(df, "Flipkart", "product_rating")

    print("Average Product Rating for queried product at amazon is : ", amz_avg_ratings)
    print("Average Product Rating for queried product at flipkart is : ", flk_avg_ratings)

    plt.figure(figsize=(15, 10))

    # Box plot
    b = sns.boxplot(data=df,
                    x="ecommerce_website",
                    y="product_rating",
                    width=0.4,
                    linewidth=2,

                    showfliers=False)
    # Strip plot
    b = sns.stripplot(data=df,
                      x="ecommerce_website",
                      y="product_rating",
                      # Colours the dots
                      linewidth=1,
                      alpha=0.4)

    b.set_ylabel("Ratings (out of 5)", fontsize=14, color="red")
    b.set_xlabel("E-Commerce Website", fontsize=14, color="red")
    b.set_title("Comparing Ratings (box-plot)", fontsize=30, c='b')

    sns.despine(offset=5, trim=True)
    # b.get_figure()



    return b


# User Defined Function : Analysing the relationship between Product's Price and its Rating
# Function that plots a scatter plot between price and ratings
def price_rat_corr(df):
    # Printing what kind of correlation is present between product price and product rating
    price_rating_corr = df.corr()['product_rating']["product_price"]

    if abs(price_rating_corr) > 0.9:
        print("Product Ratings and Product Prices are very hignly correlated. ")

    if abs(price_rating_corr) > 0.7:
        print("Product Ratings and Product Prices are hignly correlated. ")

    if abs(price_rating_corr) > 0.5:
        print("Product Ratings and Product Prices are moderately correlated. ")

    if abs(price_rating_corr) > 0.3:
        print("Product Ratings and Product Prices have low correlation. ")

    else:
        print("Product Ratings and Product Prices have very little to no correlation. ")

    fig = px.scatter(df, x="product_rating", y="product_price",
                     color="ecommerce_website", template="plotly_dark",
                     color_discrete_sequence=colors,
                     trendline="ols",
                     title="Analysing the relationship between Product's Price and Product's Ratings")

    fig.update_xaxes(showticklabels=True)

    return fig.show()






"""
Main function for vsiualising all the results
"""


def visualiseQueryReults(df, brand_present_df):
    # amazon orange, flipkart yellow
    colors = ["#FEBD69", "#F8E831"]
    customPalette = sns.set_palette(sns.color_palette(colors))

    # Change some of seaborn's style settings with `sns.set()`
    sns.set(style="ticks",  # The 'ticks' style
            rc={ "figure.facecolor": "ivory",  # Figure colour
                "axes.facecolor": "ivory"},
            palette=customPalette)  # Axes colour

    ax1 = countProductsPlot(df)
    if len(brand_present_df)==len(df):
        pass
    else:
        ax2 = countActualProductsPlot(df, brand_present_df)

    ax3 = priceCompLine(brand_present_df)
    ax4 = CompPriceBox(brand_present_df)
    ax5 = priceNamePlotly(brand_present_df)

    ax6 = CompRatingBox(brand_present_df)

    ax7 = price_rat_corr(brand_present_df)
