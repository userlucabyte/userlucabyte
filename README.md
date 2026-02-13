# PROJECT CONTEXT:
# This project aims to perform unsupervised clustering on articles
# from the French newspaper "Le Monde" in order to automatically
# group them into thematic domains such as culture, sports, politics, etc.
# This script corresponds to the DATA COLLECTION phase,
# which is the first and essential step before preprocessing,
# vectorization (e.g., TF-IDF), and clustering (e.g., K-Means).


# Import required libraries
import requests
from bs4 import BeautifulSoup
import json

# requests:
# Used to send HTTP requests and download the HTML content
# of each article page from Le Monde.
#
# BeautifulSoup:
# Parses the raw HTML and allows structured extraction of
# specific elements such as title, publication date, and text.
#
# json:
# Used to store the scraped data in structured format (JSON file),
# which will later be used for text preprocessing and clustering.

def scrape_article(url):
    """
    This function extracts structured information from a single article URL.

    It retrieves:
    - The article title
    - The publication date
    - The full article text (all paragraphs)

    The output is a dictionary that represents one document
    in our future clustering dataset.
    """

    # Send HTTP request to retrieve the webpage content
    # timeout=10 prevents the program from freezing if the server is slow
    r = requests.get(url, timeout=10)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")

    # Extract the article title
    # The title is typically contained in the <h1> tag
    # strip=True removes unnecessary whitespace
    # If the tag does not exist, we return an empty string
    
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract the publication date
    # The date is usually stored inside a <time> tag
    # The datetime attribute provides a clean structured format
    
    time_tag = soup.find("time")
    date = time_tag.get("datetime", "") if time_tag else ""

    # --------------------------------------------------------
    # Extract the article body
    # All paragraph tags <p> are collected
    # We concatenate them into one single text string
    # This raw text will later be cleaned and vectorized
    # for clustering purposes
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs)

    # Return a structured dictionary
    # Each dictionary represents one document in our corpus
    # This format is ideal for later transformation into
    # a dataframe and applying NLP techniques
    
    return {
        "url": url,
        "title": title,
        "date": date,
        "text": text
        
# Initialize an empty list to store all scraped articles
# This list will represent our raw corpus
articles = []

# Read the list of URLs from an external file (URLS.txt)
# Only lines starting with "http" are kept
# This ensures that only valid article links are processed

with open("URLS.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip().startswith("http")]

# Loop through each URL and scrape the article
# A try/except block is used to avoid stopping the process
# if one article fails (robust data collection)

for url in urls:
    try:
        article = scrape_article(url)
        articles.append(article)
        print("OK :", url)
    except Exception as e:
        # If an error occurs, we skip the article
        # This ensures the scraping continues
        print("Erreur :", url)

# Save the collected articles into a JSON file
# ensure_ascii=False preserves French characters
# indent=2 makes the file readable
#
# This JSON file will serve as the input dataset
# for the unsupervised clustering pipeline

with open("articles_raw.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

# Print the number of successfully scraped articles
# This gives a quick verification of dataset size

print("Nombre d'articles scrapés :", len(articles))


# SUMMARY IN THE CONTEXT OF UNSUPERVISED CLUSTERING:
#
# 1. This script builds the raw text corpus.
# 2. Each article becomes one document.
# 3. The collected text will later be:
#    - Cleaned (stopwords removal, lowercasing, etc.)
#    - Vectorized (e.g., TF-IDF)
#    - Clustered (e.g., K-Means, Hierarchical clustering)
#
# Without this structured data extraction phase,
# unsupervised learning would not be possible.


# SECOND STEP: TEXT CLEANING (PREPROCESSING PHASE)
#
# After collecting raw articles from "Le Monde",
# this step prepares the textual data for vectorization
# and unsupervised clustering.
#
# The objective here is to normalize and clean the text
# in order to reduce noise and improve clustering quality.


import json
import re

# clean_text FUNCTION
# This function applies basic preprocessing operations
# to standardize the textual content of each article
def clean_text(text):

    # Convert text to lowercase
    # This avoids treating the same word differently
    # (e.g., "France" and "france")
    text = text.lower()

    # Remove potential HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters, digits, and punctuation
    # Keep only alphabetic characters (including French accents)
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    return text.strip()

# Load the raw scraped articles
# This file was generated during the data collection phase
with open("articles_raw.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Create a new list to store cleaned articles
# Each document will now contain normalized text only
clean_articles = []

for art in articles:
    clean_articles.append({
        "url": art["url"],
        "text": clean_text(art["text"])
    })

# Save the cleaned dataset into a new JSON file
# This file will be used for vectorization (e.g., TF-IDF)
# before applying clustering algorithms
with open("articles_clean.json", "w", encoding="utf-8") as f:
    json.dump(clean_articles, f, ensure_ascii=False, indent=2)

# Display the number of cleaned articles
# Ensures consistency with the original dataset size
print("Articles nettoyés :", len(clean_articles))

# ROLE IN THE UNSUPERVISED CLUSTERING PIPELINE:
#
# 1. Reduces noise (punctuation, symbols, inconsistencies)
# 2. Standardizes text format
# 3. Improves quality of feature extraction
# Clean and normalized text leads to more meaningful
# document vectors and therefore better thematic clusters
# (culture, sports, politics, etc.).

# FINAL STEP: VECTORIZATION + UNSUPERVISED CLUSTERING
# This is the core of the project. After scraping and cleaning
# articles from "Le Monde", we now transform text into numerical
# representations and apply an unsupervised learning algorithm
# to automatically group articles into thematic clusters.

import json
import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Load cleaned articles
# These texts are already normalized and ready for NLP tasks

with open("articles_clean.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

texts = [a["text"] for a in articles]   # Article contents
urls = [a["url"] for a in articles]     # Corresponding URLs


# 1- TF-IDF VECTORIZATION
# Text cannot be directly processed by clustering algorithms.
# TF-IDF converts each document into a numerical vector
# representing the importance of words in the corpus.
#
# max_df=0.8 → Ignore very frequent words (too common)
# min_df=2   → Ignore very rare words
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(texts)

# 2️-K-MEANS CLUSTERING
# K-Means groups documents into k clusters
# based on similarity in vector space.
#
# k represents the number of thematic groups.
# Here we choose 5 clusters (can be tuned).
k = 5  # typically between 3 and 6 for this type of dataset

model = KMeans(n_clusters=k, random_state=42)

# fit_predict:
# - Learns cluster centers
# - Assigns each article to a cluster
clusters = model.fit_predict(X)


# 3️-CLUSTER INTERPRETATION
# Since clustering is unsupervised, clusters do not
# automatically have names.
#
# To interpret them, we extract the most important
# words (top TF-IDF weights) near each cluster center.
terms = vectorizer.get_feature_names_out()
cluster_keywords = {}

for i in range(k):
    center = model.cluster_centers_[i]

    # Get indices of top 8 most important words
    top_terms = center.argsort()[-8:]

    cluster_keywords[i] = [terms[t] for t in top_terms]


# 4️⃣ MANUAL CLUSTER LABELING
# After examining the dominant keywords,
# we manually assign semantic names to clusters.
#
# These names correspond to major domains such as:
# politics, sports, economy, culture, technology.
cluster_names = {
    0: "Politique internationale",
    1: "Sport",
    2: "Économie",
    3: "Culture et médias",
    4: "Technologie et innovation"
}


# 5️⃣ SAVE RESULTS TO CSV
# Each article is saved with:
# - its URL
# - cluster ID
# - assigned cluster name
#
# This file can be used for analysis or visualization.
with open("clusters.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "cluster_id", "cluster_name"])

    for url, c in zip(urls, clusters):
        writer.writerow([url, c, cluster_names.get(c, "Inconnu")])


# 6️⃣ VERIFICATION AND DISTRIBUTION
# Counter shows how many articles are in each cluster.
# This helps evaluate balance between thematic groups.
#
# Printing keywords helps validate interpretation.
print("Répartition des clusters :", Counter(clusters))

for i, kw in cluster_keywords.items():
    print(f"Cluster {i} :", ", ".join(kw))

# PROJECT SUMMARY (UNSUPERVISED PIPELINE):
# 1. Web scraping → build raw dataset
# 2. Text cleaning → normalize corpus
# 3. TF-IDF → convert text to vectors
# 4. K-Means → group similar articles
# 5. Keyword analysis → interpret clusters
#
# The system automatically discovers thematic structures
# in the newspaper without using predefined labels.
