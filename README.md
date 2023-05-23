# Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer
This is an nlp, web-scrapping based ML capstone project, that allows users to retrieve datasets of the products listed on flipkart and amazon for the given product. Further, Sentiment Analysis can also be performed on the comments/reviews.

---
> ## Introduction
<p align="justify">
Customer confusion has been on the uprising for the past decade, this psychological construct can be credited to the rampant e-commerce industry. In 2021, e-retail sales crossed a mammoth 5.2 trillion USD worldwide. Although the numbers can be assumed to be bloated, as a customer, the plethora of e-commerce websites to choose from has made purchasing decisions harder. Ill-informed decisions regarding buying a product can lead to mistrust in online shopping or financial disasters. It's imperative to keep the customers well-informed to keep them satisfied, it also saves ample of time while purchasing a product.
</p>

---
> ## Background
<p align="justify">
The three underlying principles of e-confusion or customer confusion are similarity confusion, overload confusion, and unclarity confusion. But, in most developing countries like India, the biggest cause of consumer swingers’ is product price, rating, reviews and of course availability. Keeping this in mind, the project is aimed at building a thorough application to help users in deciding which product to buy from, and more importantly where from. The added feature of generating a sentiment analysis of the reviews made by customers, provides an in-depth understanding of the position of customers regarding the product. Opinion mining was an agenda that was exploited during the project, it helped in determining whether the overall customer experience of the customer is positive, neutral or negative in nature.
</p>

---
> ## Problem Statement
<p align="justify">
This project attempts to combine data analytics methodologies like KDD (Knowledge Discovery Database) and comparative analysis techniques to tackle an ever-lasting conundrum, that is settling the question, which platform is the best to buy your desired product from? Considering time constraints, the comparisons were limited to Amazon and Walmart subsidiary Flipkart. The same is done by web scraping the requested products query from Amazon and Flipkart, and further web-scrapping product reviews, enabling individual product-based sentimental analysis (Amazon Only).
</p>

---
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

---
> ## Data Sources and Collection
<p align="justify">
The impromptu response nature of the project required an on-demand and customizable dataset generation, every time the user seeks a product. To tackle the complexity, we created three modules of web scraping for specific purposes and generating specific datasets.

The sources of the web scraped data were Amazon and Flipkart’s website. The collection involved a generalized algorithm of web scraping products and their information.

The reviews scrapped module went through all the comments of the product and, post data scrubbing, storing the information in a dataset. Post scrapping the reviews the sentiment analysis module created by me, added 8 columns based on a ROBERTA and VADER model sentiment scores. These scores helped in generating an aggregate opinion (polarity) representative of all the customers.
</p>

There were two main datasets that were created through the process.
- Web Scrapping Amazon (based on query (product given by client)
- Web Scrapping Flipkart (based on query (product given by client)

The above datasets are merged and the final product dataset created with the web scraped data had the following variables:
1. query_searched
2. ecommerce_website
3. product_name
4. product_price
5. product_rating
6. rating_count
7. product_image_url
8. product_url

__Sample Product Listing (Amazon + Flipkart Dataset)__ <br><br>
<img width="1132" alt="image" src="https://github.com/shreyansh-2003/Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer/assets/105413094/6acf587e-87fb-439c-8707-0a2d48fc44a8">
<br>

Web Scrapping Comments (based on the product selected from the above dataset) (Amazon Only).

The comments scrapped dataset had the following variables:
1. customer_name
2. review_date
3. review_title
4. review_ratings
5. review_content
6. review_size
7. sentiment
8. roberta_sentiment
9. vader_sentiment
10. roberta_neg
11. roberta_neu
12. roberta_pos
13. vader_neg
14. vader_pos
15. vader_neu


__Sample Comments/Review for a Condom (Amazon)__<br><br>
![image](https://github.com/shreyansh-2003/Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer/assets/105413094/7f5618e3-0c1d-48fd-8b63-1f7035d55948)
<br>


---

> ## Methodology
<p align="justify">
The process and methodology involved during the project were a robust and in-depth exploration of various components and fragments of KDD and data analytics.
</p>

1. Understanding requirements to handle the Problem Statement.
   - This first vital step made me realize the complexity of the project. Web-scrapping using Python Libraries had to be learnt from scratch using tutorials and library documentation.
2. Designing the architecture of the program
   - From the onset, it was understood that the project required an elaborate backend and a sophisticated design to help integrate it with the finesse of the front end. This prompted me to create 3 separate modules and import it into the final file as .py finals. Using concepts from OOPs would result in a simplified and easy to traceback software.

The backend modules include:
- __webscraping.py module__:
  - Web Scrapping Amazon Class (based on query (product given by the client)
  - Web Scrapping Flipkart Class (based on query (product given by the client)

- __ProjectVisualizations.py module__:
  - Consists of various functions that help in plotting the amazon vs Flipkart data.
  - Using Plotly, seaborn, and matplotlib to come up with 17 plots for a website vs website analysis.

- __ProductReviewAnalysis.py module__:
  - This file contained was the main file for performing a sentiment analysis upon reviews/comments on a product (currently limited to amazon)
  - Using ROBERTA Model (NLP)
  - Using VADER Model (NLP)
  - A weighted 60:40 approach (Roberta : Vader) to obtain a final sentiment value.
  - 18 Visually Appealing Plots

- __Inspecting elements of static HTML pages of Amazon and Flipkart and Scrapping based on trial and error__
  - The web scraping was done in tandem with the requests module and BeautifulSoup module, this involved requesting the search page based on the user’s query. The page retrieved then had to be scouted for required information. This was done with basic knowledge of the inbuilt methods of the libraries being used and on a trial-and-error basis. It was a tedious process as it required checking where the elements were present on the actual page and comparing it in my script.
  
- __Cleaning data and transforming data while it’s being scrapped__
  - A major difference between traditional EDAs and models created by me in the past and this project was, I had to clean and make the data apt while it was being scrapped and before inserting it into the dataset. This involved using regex and string manipulation functions to extract relevant information and convert it into a usable format.


---

> ## Apendix

<br>

#### Sample Product Listing View

<img width="625" alt="Product Listing" src="https://github.com/shreyansh-2003/Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer/assets/105413094/0ee79a7f-6892-45b7-953a-e13be7940178">

<br>


#### Sample Product Listing comparative analysis through visualisations
<img width="1070" alt="image" src="https://github.com/shreyansh-2003/Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer/assets/105413094/a20ba0cc-936b-4f8b-b701-09beb5294f50">
<br>


#### Sample Reviews and NLP analysis through visualisations

![image](https://github.com/shreyansh-2003/Amazon-Flipkart-Product-Scraper-and-Sentiment-Analyzer/assets/105413094/484e8640-568d-4076-807e-6a0b08bdbed4)

<br>

---
