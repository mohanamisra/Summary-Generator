# AI/ML Product Review Analysis

We have built an Artificial Intelligence model that provides a thorough review of any valid tech product that the user chooses to enter.

There were two main components involved in building this. The web scraper and the AI model itself.

The web scraper has been built using PRAW, which scrapes the data from all relevant subreddits based on which product name the user enters.

The AI model analyses the output csv of the scraping to generate the reviews and classify them as Positive and Negative.

### Project Goals:
1. To help businesses gauge how well their product is performing in the market
2. To help organisations get an idea of what their target demographic thinks of their product
3. To help product-designers understand what direction the market points to
4. To help users make smart decisions before purchase of a product
5.

### How to Use the Project:
1. Installations:
```
rstpip install scikit-learn pandas numpy textblob
pip install flask
pip install python-dotenv
pip install nltk
```
2. Clone the repository:
```
git clone  git@github.com:harshita604/Summary-Generator.git
```
3. Navigate to the repository in the terminal:
```
cd Summary-Generator
```
4. Run the project:
```
python app.py
```  


### UI Preview:
[to be added]