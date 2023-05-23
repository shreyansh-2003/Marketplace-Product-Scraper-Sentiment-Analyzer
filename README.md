# Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer
This is an nlp, web-scrapping based ML capstone project, that allows users to retrieve datasets of the products listed on flipkart and amazon for the given product. Further, Sentiment Analysis can also be performed on the comments/reviews.

> ## Introduction
Customer confusion has been on the uprising for the past decade, this psychological construct can be credited to the rampant e-commerce industry. In 2021, e-retail sales crossed a mammoth 5.2 trillion USD worldwide. Although the numbers can be assumed to be bloated, as a customer, the plethora of e-commerce websites to choose from has made purchasing decisions harder. Ill-informed decisions regarding buying a product can lead to mistrust in online shopping or financial disasters. It's imperative to keep the customers well-informed to keep them satisfied, it also saves ample of time while purchasing a product.


> ## Background
The three underlying principles of e-confusion or customer confusion are similarity confusion, overload confusion, and unclarity confusion. But, in most developing countries like India, the biggest cause of consumer swingers’ is product price, rating, reviews and of course availability. Keeping this in mind, the project is aimed at building a thorough application to help users in deciding which product to buy from, and more importantly where from. The added feature of generating a sentiment analysis of the reviews made by customers, provides an in-depth understanding of the position of customers regarding the product. Opinion mining was an agenda that was exploited during the project, it helped in determining whether the overall customer experience of the customer is positive, neutral or negative in nature.

> ## Problem Statement
This project attempts to combine data analytics methodologies like KDD (Knowledge Discovery Database) and comparative analysis techniques to tackle an ever-lasting conundrum, that is settling the question, which platform is the best to buy your desired product from? Considering time constraints, the comparisons were limited to Amazon and Walmart subsidiary Flipkart. The same is done by web scraping the requested products query from Amazon and Flipkart, and further web-scrapping product reviews, enabling individual product-based sentimental analysis (Amazon Only).

> ## REPOSITORY STRUCTURE

The __Src Scripts__ folder holds all the .py files, as Classes and user defined functions that can be called for their specific functionality.:
1. __ProductReviewAnalysis.py__: NLP, comments pre-processing, sentiment analysis using VADER polarity score.
2. __ProjectVisualisation.py__: The python script has functions that plots over 30 different kind of visualisations for both reviews and products.
3. __ecom_prod_scraper.py-__: FlipkartScraper and AmaonScraper classes with functions that help retrieve products from Amazon and Flipkart.
4. __amz_comments_scraper.py__: This helps in scrapping amazon comments for a given prompt (product).

The ```sample_driver_code.ipynb``` holds a small sample code on how to create a dataset for a product (query), that has listings from amazon and flipkart.

```bash
|____Src Scripts
| |____ProductReviewAnalysis.py_
| |____ProjectVisualisations.py
| |____ecom_prod_scraper.py
| |____amz_comments_scraper.py
| |____sample_driver_code.ipynb
|____Sample Datasets Webscarapped
| |____sample_comments_scrapped.csv
| |____sample_products_scrapped.csv
|____Sample Screenshots
| |____Products
| |____NLP Analysis
|____Website (Flask)
| |____trial.html
| |____websiteTrial.py
| |____styles.css
| |____ProdPage.py
| |____templates
| | |____index.html
| | |____reviews_analysis.html
| | |____static
```
