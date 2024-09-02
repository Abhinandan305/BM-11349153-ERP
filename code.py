#!/usr/bin/env python
# coding: utf-8

# # IMPORTING REQUIRED LIBRARIES
# ### All the packages needed for data pre-processing, machine learning and data visualization have been installed

# In[1]:


import pandas as pd
import re
import nltk
from langdetect import detect
import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import  pos_tag
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import string
from collections import Counter
from wordcloud import WordCloud
import numpy as np

#Installing Vader sentiment analysis tool
get_ipython().system('pip install vaderSentiment')

# Downloading stopwords and punctuation library for preprocessing
nltk.download('stopwords')
nltk.download('punkt')


# # IMPORTING REVIEWS DATA FOR GREATER MANCHESTER

# In[2]:


# Loading the reviews dataset

df = pd.read_csv('/Users/abhinandandas/Downloads/reviews-manc.csv')


# Displaying the top 5 records of the reviews dataframe
df.head(5)


# # DATA PRE-PROCESSING
#  
#  
# ### A total of 188757 reviews were present
# #### Missing values have been removed
# #### Contractions have been expanded.
# #### Comments less than 30 charecters have been removed since they don't carry much information.
# #### Only English reviews are kept.
# #### Non-ASCII charecters are converted to ASCII
# #### Date formats have been parsed
# #### Filtering reviews and keeping only the reviews from 2019
# #### Converting all comments to lowercase to maintain consistency
# #### Adding a column to track number of reviews  per year 
# #### Adding a column to count number of words, sentences and charecters in reviews
# #### Replacing certain words to ensure data consistency
# #### Removing br tags, angular brackets and other html tags
# #### Replacing certain words to ensure data consistency

# In[3]:


# Dropping missing values
df.dropna(inplace=True)

# Removing reviewer_name column from the dataset 
df.drop(columns={'reviewer_name'}, inplace=True)

# Custom contraction map for expanding contracted words
CONTRACTION_MAP = {
    "can't": "cannot", "won't": "will not", "i'm": "i am", "i'd": "i would", "i've": "i have",
    "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is", "they're": "they are",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "didn't": "did not",
    "don't": "do not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "shouldn't": "should not",
    "wouldn't": "would not", "couldn't": "could not", "mightn't": "might not", "mustn't": "must not"
}

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match.lower())
        expanded_contraction = match[0] + expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Expanding contractions using the function
df['comments'] = df['comments'].apply(lambda x: expand_contractions(x))

# Adding a column for review length and filtering out short reviews
df['ReviewLength'] = df['comments'].apply(lambda x: len(x))
df = df[df.ReviewLength >= 30]

# Detect language and keep only English reviews
df['Lang'] = [detect(elem) if len(elem) > 50 else 'no' for elem in df['comments']]
df = df[df['Lang'] == 'en']

# Converting non-ASCII to ASCII
df['comments'] = df['comments'].apply(lambda x: unidecode.unidecode(x))

# Parsing date column for different formats
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except ValueError:
            return pd.NaT

df['date'] = df['date'].apply(parse_date)

# Filtering reviews from 2019 onwards
df = df[df['date'].dt.year >= 2019]

# Converting comments to lowercase for data consistency
df['comments'] = df['comments'].str.lower()

# Dropping missing values and unnecessary columns if any
df.dropna(inplace=True)

# Adding a column to track number of reviews  per year 
df['Year'] = pd.DatetimeIndex(df['date']).year
ReviewDistribution = df.groupby(['Year']).comments.count().reset_index()

# Cleaning comments and removing blank spaces
df["comments"].replace(" '","'", inplace=True)
df["comments"].replace("' s","'s", inplace=True)

# Adding columns for sentence, word, and character counts
df['Number of Sentences'] = df['comments'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
df['Number of words'] = df['comments'].apply(lambda x: len(x.split()))
df['Number of Characters'] = df['comments'].str.len()

# Defining a dictionary for word replacements to ensure data consistency
replacement_dict = {
    r'hosts': 'host',r'comfy': 'comfortable', r'apartments': 'apartment', 
    r'flat': 'apartment', r'house': 'home',
    r'host place': 'home', r'air bnb': 'airbnb',
    r'airb&b': 'airbnb', r'en suite': 'en-suite',
    r' ensuite': 'en-suite', r'city centre': 'city-centre', r'city center': 'city-centre',
    r'town centre': 'town-centre', r'town center': 'town-centre', r'&': 'and',
    r'accomodating': 'accommodating', r'amazig': 'amazing', r'amazng': 'amazing',
    r'avaiable': 'available', r'welcomming': 'welcoming', r'wondeful': 'wonderful',
    r'wounderful': 'wonderful', r'knowledgable': 'knowledgeable', r'v, helpful': 'helpful',
    r'quick, responding': 'responding', r'confortable': 'comfortable', r'comfotable': 'comfortable',
    r'fab': 'fabulous', r'comfortable, sitting': 'comfortable', r'extra, comfortable': 'comfortable',
    r'lovey': 'lovely', r'beatiful': 'beautiful', r'beautifull': 'beautiful',
    r'icnredible': 'incredible', r'helful': 'helpful', r'relly': 'really'
}

# Function to replace terms based on the dictionary 
def replace_terms(text, replacements):
    for term, replacement in replacements.items():
        text = re.sub(rf'\b{term}\b', replacement, text, flags=re.IGNORECASE)
    return text

# Applying the function to the 'comments' column
df['comments'] = df['comments'].apply(lambda x: replace_terms(x, replacement_dict))

# Replacing br tags and angular brackets present in reviews
chars = ['br/', '<', '>']

for char in chars:
    df['comments'] = df['comments'].str.replace(char, '')

# Defining and storing stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from comments
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Applying the function to the 'comments' column
df['comments_clean'] = df['comments'].astype(str).apply(remove_stopwords)

# Display the first few rows of the dataframe to check the new column
df[['comments', 'comments_clean']].head()


# # IMPORTING LISTINGS DATA FOR GREATER MANCHESTER

# In[4]:


Listings = pd.read_csv("/Users/abhinandandas/Downloads/listings.csv")

#Renaming the "id" column to listing_id
Listings.rename(columns={'id':'listing_id'}, inplace=True)


# # MERGING BOTH THE DATAFRAMES

# In[5]:


df = df.merge(Listings[['listing_id', 'host_name', 'neighbourhood', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']], on='listing_id', how='inner')
df.dtypes
df.head(5)


# # GENERATING WORD CLOUD OF MOST MENTIONED WORDS

# In[6]:


# Joining all comments into a single string
wordlist = ' '.join(df['comments_clean']).split()

# Removing punctuation from the word list
wordlist_clean = [word.strip(string.punctuation) for word in wordlist if word.strip(string.punctuation) != '']

# Storing the frequency of each word
counts = Counter(wordlist_clean)

# Generating word cloud
wordcloud = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(counts)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Common Words')
plt.show()

# Displaying the most common words along with their counts
words = counts.most_common(20)
print(words)


# # ANONYMIZING DATA
# 
# 

# In[ ]:


# Generating unique identifiers for each host name
host_name_to_id = {name: f"Host_{i+1}" for i, name in enumerate(df['host_name'].unique())}

# Anonymizing the host_name column
df['host_name'] = df['host_name'].map(host_name_to_id)

# Verifying the changes
print(df.head())


# # EXPLORATORY DATA ANALYSIS

# ### COMPARING PRICE WITH NEIGHBOURHOOD

# In[7]:


# Grouping by neighbourhood and calculating the average price
avg_price = df.groupby('neighbourhood')['price'].mean()

# Plot the bar chart
plt.figure(figsize=(9, 6))
avg_price.plot(kind='bar', color='purple')
plt.title("Average Price by Neighbourhood")
plt.xlabel("Neighbourhood")
plt.ylabel("Average Price")
plt.grid(True)
plt.show()


# ### FINDING TOTAL NUMBER OF HOSTS AND LISTINGS

# In[10]:


# Grouping listings per host
listings_per_host = df.groupby('host_name')['listing_id'].count()

# Calculating the total number of hosts and  listings
total_hosts = len(listings_per_host)
total_listings = listings_per_host.sum()

# Creating a table to show the results
summary_table = pd.DataFrame({
    'Total Hosts': [total_hosts],
    'Total Listings': [total_listings]
})

print(summary_table)


# ### NEIGHBOURHOOD WITH MOST LISTINGS

# In[11]:


# Grouping 'neighbourhood' with number of unique listings
listings_per_neighbourhood = df.groupby('neighbourhood')['listing_id'].nunique()

# Finding the neighbourhood with the most listings
most_listings_neighbourhood = listings_per_neighbourhood.idxmax()
most_listings_count = listings_per_neighbourhood.max()

print(f"Neighbourhood with the most listings: {most_listings_neighbourhood} ({most_listings_count} listings)")


# ### NEIGHBOURHOOD WITH LEAST NUMBER OF LISTINGS

# In[12]:


# Similarly, finding the neighbourhood with the least listings
least_listings_neighbourhood = listings_per_neighbourhood.idxmin()
least_listings_count = listings_per_neighbourhood.min()

print(f"Neighbourhood with the least listings: {least_listings_neighbourhood} ({least_listings_count} listings)")


# ### COUNTING NUMBER OF PEOPLE WHO RETURENED TO THE SAME PROPERTY

# In[13]:


# Counting the number of reviews per reviewer for each listing
df['is_returner'] = df.groupby(['reviewer_id', 'listing_id'])['id'].transform('count') > 1

# Converting boolean to integer (1 for returners, 0 for non-returners)
df['is_returner'] = df['is_returner'].astype(int)

#  number of returners and non-returners
returner_counts = df['is_returner'].value_counts()


# In[18]:


# Data for the plot
labels = ['Returners', 'Non-Returners']
counts = [returner_counts.get(1, 0), returner_counts.get(0, 0)]

plt.figure(figsize=(9, 6))
bars = plt.bar(labels, counts, color=['blue', 'yellow'])

# Adding the count labels on top of  bars
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.5, int(val), ha='center', va='bottom')

#  x-axis label
plt.xlabel('Type of Reviewer')

#  y-axis label
plt.ylabel('Number of Reviewers')

plt.title('Number of Returners vs Non-Returners')
plt.show()


# ### REVIEW LENGTH DISTRIBUTION

# In[19]:


# Review length distribution

plt.figure(figsize=(10, 6))
sns.histplot(df['ReviewLength'], bins=30)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()


# ### REVIEW TREND OVER THE YEARS

# In[20]:


# Review trend over the years
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)
df['review_count'] = 1
monthly_reviews = df['review_count'].resample('M').sum()
monthly_reviews.plot(figsize=(12, 6))
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.show()


# # PERFORMING SENTIMENT ANALYSIS USING VADER

# In[21]:


# Initializing Sentiment Intensity Analyzer
v = SentimentIntensityAnalyzer()

# Function to analyze sentiment of each sentence
def analyze_sentiment(text):
    sentences = sent_tokenize(text)
    sentiment_seq = []
    for sentence in sentences:
        sentiment_score = v.polarity_scores(sentence)['compound']
        if sentiment_score >= 0.05:
            sentiment_seq.append('Positive')
        elif sentiment_score <= -0.05:
            sentiment_seq.append('Negative')
        else:
            sentiment_seq.append('Neutral')
    return sentiment_seq

# Applying sentiment analysis function to the comments
df['SentimentSequence'] = df['comments_clean'].apply(analyze_sentiment)


# In[22]:


df


# ### COUNT OF SENTIMENTS IN REVIEWS

# In[23]:


# Function to convert strings to lists 
def con_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)  
        except (SyntaxError, ValueError):
            return val  
    return val

# Applying the conversion function
df['SentimentSequence'] = df['SentimentSequence'].apply(con_eval)

# Ensuring sentiment data are in list format
all_sentiments = [sentiment for sublist in df['SentimentSequence'] if isinstance(sublist, list) for sentiment in sublist]

# Creating a new DataFrame for sentiment counts
sentiment_df = pd.DataFrame(all_sentiments, columns=['Sentiment'])

# Counting occurrences of each sentiment
sentiment_counts = sentiment_df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

print(sentiment_counts)


# In[24]:


# Visualising the results
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, palette='viridis')

# Annotating the bars with the counts
for index, row in sentiment_counts.iterrows():
    bar_plot.text(index, row.Count, row.Count, color='black', ha="center")

plt.title('Count of Sentiments in Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# # MOST MENTIONED WORDS IN POSITIVE REVIEWS

# In[25]:


# Filtering for only positive reviews
positive_reviews = df[df['SentimentSequence'].apply(lambda x: 'Positive' in x if isinstance(x, list) else False)]['comments']

# Tokenizing the words in  positive reviews and removing stopwords
stop_words = set(stopwords.words('english'))
words = []
for review in positive_reviews:
    tokens = nltk.word_tokenize(review.lower())
    filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
    words.extend(filtered_words)

# Generating word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # ANALYSING SENTIMENT WITH PRICE AND ROOM TYPE

# In[26]:


# Aggregating Sentiment Sequence to a single label (Positive, Negative, Neutral)
def get_overall_sentiment(sentiments):
    if sentiments.count('Positive') > sentiments.count('Negative'):
        return 'Positive'
    elif sentiments.count('Negative') > sentiments.count('Positive'):
        return 'Negative'
    else:
        return 'Neutral'

df['OverallSentiment'] = df['SentimentSequence'].apply(get_overall_sentiment)

# Plotting Sentiment vs. Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallSentiment', y='price', data=df)
plt.title('Sentiment vs. Price')
plt.show()

# Plotting Sentiment vs. Room Type
plt.figure(figsize=(10, 6))
sns.countplot(x='room_type', hue='OverallSentiment', data=df)
plt.title('Sentiment vs. Room Type')
plt.show()


# # NEIGHBOURHOOD WITH HIGHEST NEGATIVE REVIEWS

# In[27]:


# Creating a separate dataframe to include only reviews with negative sentiments
negative_sentiment_df = df[df['OverallSentiment'] == 'Negative']

# Grouping 'neighbourhood' by count of negative reviews
negative_neighbourhoods = negative_sentiment_df.groupby('neighbourhood').size().reset_index(name='NegativeReviewCount')

# Sorting the count in descending order
negative_neighbourhoods = negative_neighbourhoods.sort_values(by='NegativeReviewCount', ascending=False)

# Displaying neighborhoods with the most negative reviews
print(negative_neighbourhoods)


# In[28]:


plt.figure(figsize=(10, 6))
sns.barplot(x='NegativeReviewCount', y='neighbourhood', data=negative_neighbourhoods, palette='Reds_r')
plt.title('Neighbourhoods with Most Negative Sentiments')
plt.xlabel('Number of Negative Reviews')
plt.ylabel('Neighbourhood')
plt.show()


# # SENTIMENT TREND FOR A SPECIFIC LISTING

# In[35]:


#  Filtering the dataframe for a specific listing (e.g., listing_id = 157612)
listing_id = 157612
listing_reviews = df[df['listing_id'] == listing_id]

# Flattening Sentiment Sequences into individual sentiments with corresponding dates
sentiment_trend = []

for index, row in listing_reviews.iterrows():
    for sentiment in row['SentimentSequence']:
        sentiment_trend.append({'date': index, 'sentiment': sentiment})  
        
#  Dataframe for sentiment trend
sentiment_df = pd.DataFrame(sentiment_trend)

#  Converting sentiment labels into numerical values
sentiment_df['sentiment_score'] = sentiment_df['sentiment'].map({
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
})

# Converting the index to datetime for proper plotting
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Using a moving average for smoothing the sentiment trend
sentiment_df['rolling_mean'] = sentiment_df['sentiment_score'].rolling(window=3, min_periods=1).mean()

# Creating the plot
plt.figure(figsize=(12, 8))
plt.plot(sentiment_df['date'], sentiment_df['rolling_mean'], marker='o', linestyle='-', linewidth=2, markersize=6, color='blue')
plt.fill_between(sentiment_df['date'], sentiment_df['rolling_mean'], color='blue', alpha=0.1)

plt.title(f'Sentiment Trend for Listing {listing_id}')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.ylim(-1.1, 1.1)  # Adjusting the y-axis limits for better visualization
plt.grid(True)
plt.show()


# # TRACING SENTIMENT SHIFTS IN REVIEWS (RQ1)

# In[36]:


# Function for tracking sentiment shift patterns
def identify_pattern(sentiments):
    if len(sentiments) < 2:
        return "Single Sentiment"
    first = sentiments[0]
    last = sentiments[-1]
    return f"{first} → {last}"

# Applying function to identify patterns
df['SentimentShifts'] = df['SentimentSequence'].apply(identify_pattern)

# Displaying frequency of each pattern
shift_count = df['SentimentShifts'].value_counts().reset_index()
shift_count.columns = ['Sentiment Sequence', 'Frequency']


# In[37]:


shift_count


# # HEATMAP FOR SENTIMENT TRANSITIONS

# In[38]:


# Initializing the transition dictionary
sentiments = ['Positive', 'Neutral', 'Negative']
transitions = {pair: 0 for pair in product(sentiments, repeat=2)}

# Counting transitions
for sequence in df['SentimentSequence']:
    for current, next_one in zip(sequence, sequence[1:]):
        transitions[(current, next_one)] += 1

# Converting transitions to a DataFrame for the heatmap
transitions_df = pd.DataFrame(
    list(transitions.values()), 
    index=pd.MultiIndex.from_tuples(transitions.keys()), 
    columns=['Count']
).unstack().fillna(0)['Count']

# Creating the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(transitions_df, annot=True, cmap='coolwarm', fmt='d')
plt.title("Heatmap of Sentiment Transitions")
plt.xlabel("To Sentiment")
plt.ylabel("From Sentiment")
plt.show()


# # SENTIMENT SHIFTS IN A REVIEW

# In[39]:


# Converting sentiment sequence to numerical values for plotting
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['SentimentNumerical'] = df['SentimentSequence'].apply(lambda x: [sentiment_map[s] for s in x])

# Plotting sentiment shift for the first few reviews
plt.figure(figsize=(14, 8))
for i in range(3):  
    plt.plot(df['SentimentNumerical'].iloc[i], label=f"Review {i+1}")

plt.axhline(0, color='grey', linestyle='--')
plt.title("Sentiment Shift Across Sentences in Reviews")
plt.xlabel("Sentence Number")
plt.ylabel("Sentiment")
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.legend()
plt.show()


# In[41]:


df


# # OVERALL SENTIMENT DISTRIBUTION

# In[42]:


# Calculating the distribution of the 'OverallSentiment' 
sentiment_distribution = df['OverallSentiment'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ff9999'])
plt.title('Overall Sentiment Distribution')
plt.show()


# # FINDING MOST MENTIONED WORDS IN NEGATIVE REVIEWS

# In[66]:


# Filtering data for negative reviews
negative_reviews = df[df['OverallSentiment'] == 'Negative']

# Combining all negative reviews into a single string
all_negative_text = " ".join(negative_reviews['comments'].astype(str).tolist())

# Generating the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_negative_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Negative Reviews')
plt.show()


# # READING THE CUSTOM DICTIONARY FILE FOR ASPECT EXTRACTION

# In[46]:


# Setting the input and output paths
path = '/Users/abhinandandas/Downloads/'
outpath = '/Users/abhinandandas/Downloads/RESULTS'

# Loading the custom word list for pattern matching
list1 = pd.read_excel(path +'WordList-2.xls', sheet_name='Amenities')
Amenities = list1.iloc[:, 0]
pattern = '|'.join(Amenities)


# In[47]:


# Function to search for a pattern in a string
def pattern_searcher(search_str: str, search_list: str):
    search_obj = re.search(search_list, search_str)
    if search_obj:
        return_str = search_str[search_obj.start(): search_obj.end()]
    else:
        return_str = 'NA'
    return return_str


# # EXTRACTING AMENITIES

# In[48]:


# Pattern search for 'Amenities'
df['Amenities'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern))
AmenitiesDistributionDF = df.groupby(['Amenities']).Amenities.count().rename("Frequency").reset_index()
df['Amenities'] = np.where(df['Amenities'] == "NA", '0', '1')
print(df['Amenities'].describe())


# # EXTRACTING ENVIRONMENT

# In[49]:


# Pattern search for 'Environment'
list2 = pd.read_excel(path + 'WordList-2.xls', sheet_name='Environment')
Environment = list2.iloc[:, 0]
lst = [' ' + x + ' ' for x in Environment]
pattern2 = '|'.join(lst)
df['Environment'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern2))
EnvironmentDistributionDF = df.groupby(['Environment']).Environment.count().rename("Frequency").reset_index()
df['Environment'] = np.where(df['Environment'] == "NA", '0', '1')
print(df['Environment'].describe())


# # EXTRACTING OUTCOME

# In[50]:


# Pattern search for 'Outcome'
list3 = pd.read_excel(path + 'WordList-2.xls', sheet_name='Outcome')
Outcome = list3.iloc[:, 0]
pattern3 = '|'.join(Outcome)
df['Outcome'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern3))
OutcomeDistributionDF = df.groupby(['Outcome']).Outcome.count().rename("Frequency").reset_index()
df['Outcome'] = np.where(df['Outcome'] == "NA", '0', '1')


# # EXTRACTING WORTH

# In[51]:


# Pattern search for 'Worth'
list4 = pd.read_excel(path + 'WordList-2.xls', sheet_name='Worth')
Worth = list4.iloc[:, 0]
pattern4 = '|'.join(Worth)
df['Worth'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern4))
WorthDistributionDF = df.groupby(['Worth']).Worth.count().rename("Frequency").reset_index()
df['Worth'] = np.where(df['Worth'] == "NA", '0', '1')
print(df['Worth'].describe())


# # EXTRACTING RECOMMEND

# In[52]:


# Repeat the same process for 'Recommend'
list5 = pd.read_excel(path + 'WordList-2.xls', sheet_name='Recommend')
Recommend = list5.iloc[:, 0]
pattern5 = '|'.join(Recommend)
df['Recommend'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern5))
RecommendDistributionDF = df.groupby(['Recommend']).Recommend.count().rename("Frequency").reset_index()
df['Recommend'] = np.where(df['Recommend'] == "NA", '0', '1')
print(df['Recommend'].describe())


# # EXTRACTING NEIGHBOURHOOD

# In[53]:


# Repeat the same process for 'Neighbourhood'
list6 = pd.read_excel(path + 'WordList-2.xls', sheet_name='Neighbourhood')
Location = list6.iloc[:, 0]
pattern = '|'.join(Location)
df['Neighbourhood'] = df['comments'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern))
NeighbourhoodDistributionDF = df.groupby(['Neighbourhood']).Neighbourhood.count().rename("Frequency").reset_index()
df['Neighbourhood'] = np.where(df['Neighbourhood'] == "NA", '0', '1')
print(df['Neighbourhood'].describe())


# # CALCULATING ASPECT FREQUENCY USING SEQUENCE MINING (RQ2)

# In[54]:


# Defining the aspects to be analyzed
aspects = ['Amenities', 'Environment', 'Outcome', 'Worth', 'Recommend', 'Neighbourhood']

# Function to assign scores based on mention order
def assign_aspect_priority(row):
    # Initialize scores for each aspect
    scores = {aspect: 0 for aspect in aspects}
    
    # List the aspects mentioned in this review
    mentioned_aspects = [aspect for aspect in aspects if row[aspect] == 1]
    
    # Assign scores based on the order of mention
    if mentioned_aspects:
        total_mentions = len(mentioned_aspects)
        for i, aspect in enumerate(mentioned_aspects):
            # Score = total mentions - position (i.e., higher score for earlier mentions)
            scores[aspect] = total_mentions - i
    
    return pd.Series(scores)

# Applying function to each row
df_aspect_scores = df.apply(assign_aspect_priority, axis=1)

# Adding up the scores for each aspect across all reviews
aspect_priority = df_aspect_scores.sum()
print(aspect_priority)


# In[55]:


# Checking the distribution of 0s and 1s in the aspect columns
aspect_counts = df[aspects].sum()
print(aspect_counts)


# In[56]:


# Converting the aspect columns to numeric (integers)
df[aspects] = df[aspects].apply(pd.to_numeric, errors='coerce')

# Checking distribution of 0s and 1s in the aspect columns for counting occurences
aspect_counts = df[aspects].sum()
print(aspect_counts)


# In[57]:


# Visualising the frequency of aspects using a bar chart
plt.figure(figsize=(10, 6))
aspect_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')

plt.title('Frequency of Aspects in Reviews')
plt.xlabel('Aspect')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Displaying the frequencies and priority in descending order
print("Aspect Priority Based on Frequency in Reviews:")
print(aspect_counts.sort_values(ascending=False))


# # POSITIONAL ANALYSIS OF ASPECTS IN A REVIEW TO UNDERSTAND ITS IMPORTANCE OR PRIORITY (RQ2)

# In[58]:


aspects = ['Amenities', 'Environment', 'Outcome', 'Worth', 'Recommend', 'Neighbourhood']

# Function to categorize the position of each aspect in a review
def categorize_position(row):
    positions = {aspect: 'None' for aspect in aspects}
    total_aspects_mentioned = sum(row[aspects])
    
    if total_aspects_mentioned == 0:
        return positions
    
    # Assigning position based on the order of appearance
    aspect_indices = [i for i, aspect in enumerate(aspects) if row[aspect] == 1]
    for i in aspect_indices:
        if i < len(aspect_indices) / 3:
            positions[aspects[i]] = 'Beginning'
        elif i < 2 * len(aspect_indices) / 3:
            positions[aspects[i]] = 'Middle'
        else:
            positions[aspects[i]] = 'End'
    
    return positions

# Applying function to categorize positions
df_positions = df.apply(categorize_position, axis=1, result_type='expand')

# Counting occurrences of each position for each aspect
position_counts = df_positions.apply(pd.Series.value_counts).fillna(0)

# Calculating percentages for plotting
position_percentages = position_counts.div(position_counts.sum(axis=1), axis=0) * 100
print(position_percentages)


# In[59]:


# Creating a stacked bar chart to show the position of aspects in reviews
position_percentages.T.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightblue', 'red', 'green','purple'])

plt.title('Position of Aspects Across Reviews (Percentage)')
plt.xlabel('Aspect')
plt.ylabel('Percentage of Reviews')
plt.legend(title='Position in Review', loc='upper right')
plt.grid(axis='y')


plt.show()


# In[60]:


df.to_csv(outpath+'DF2019-23.csv')


# # BREAKING THE REVIEWS INTO SENTENCES AND APPLYING POS TAGGING

# In[61]:


df=pd.read_csv(outpath+'DF2019-23.csv')
df.dtypes

# Changing reviews to multiple rows 
sentences = []
for row in df.itertuples():
    for sentence in sent_tokenize(row[5]):
        sentences.append((row[2],row[4], sentence))
new_df = pd.DataFrame(sentences, columns=['ID','Reviewer_ID', 'SENTENCE'])

new_df['SentenceLenght']=new_df['SENTENCE'].apply(lambda x: len(x))

new_df=new_df[new_df.SentenceLenght> 3]### remove rows with length less than 3

new_df.dtypes

new_df['POSTags'] = nltk.pos_tag_sents(new_df['SENTENCE'].apply(word_tokenize).tolist())
new_df['POSTags'].head(10)


# # SPECIFYING GRAMMAR RULES TO EXTRACT NOUN PHRASES FROM SENTENCES

# In[63]:


import string
import gensim
from string import punctuation

grammar = r"""
  NP: {<JJ><NN.*>+}               # Chunk JJ followed by NN
  NP1: {<NN.*>+<JJ>}
  NP2: {<NN.*>+<JJ><JJ>*}
  NP3:{<NN.*>+<VB.*>*<JJ><JJ>*}
  NP4:{<NN.*>+<VB.*>*<RB.*>*<JJ><JJ>*}
  NP5: {<PRP$><NN.*>+<JJ><JJ>*}
  NP6:{<NN.*>+<VBZ>*<RB>*<JJ><JJ>*}
  NP7:{<JJ.*><NN.*><NN.*>+} 
  NP8:{<JJ.*>+<NN.*><NN.*>+} 
  NP9:{<JJ.*NN.*><NN.*>+}
  """
  
parser = nltk.RegexpParser(grammar)

noun_phrases = []
for sentence in new_df['SENTENCE']:
    
    # Normalizing the text
    sentence = ''.join(c for c in sentence if not c.isdigit())
    sentence = ''.join(c for c in sentence if c not in punctuation)
    
    # Getting individual words
    words = nltk.tokenize.word_tokenize(sentence)
    
    # POS Tagging
    pos_tokens = nltk.pos_tag(words)
    
    # Parsing the tagged words
    parsed_sentence = parser.parse(pos_tokens)
    
    for chunk in parsed_sentence.subtrees():
        # Finding the NP subtrees - these are noun phrases
        if chunk.label() == 'NP' or chunk.label() == 'NP1' or chunk.label() == 'NP2' or chunk.label() == 'NP3' or chunk.label() == 'NP4' or chunk.label() == 'NP5' or chunk.label() == 'NP6':
            
            # Assembling the phrase from its constituent words
            noun_phrase = []
            for word in chunk:
                noun_phrase.append(word[0])
                
            # Adding the phrase to the list of noun phrases found in the document
            noun_phrases.append(' '.join(noun_phrase))

# Printing the extracted noun phrases

print('\nNoun Phrases\n------------------')
for phrase in noun_phrases:
    print(phrase)


# # CALCULATING NOUN PHRASE FREQUENCY

# In[64]:


Test=nltk.FreqDist(noun_phrases)
Test1=pd.DataFrame(list(Test.items()), columns = ["Word","Frequency"])
Test1.to_csv(outpath+"nounphraseFrequency.csv")

# Creating a regex pattern to join all the noun phrases and for calculating frequency
mylist=Test1.iloc[:,0]
pattern = '|'.join(mylist)

# Function to search for patterns in reviews
def pattern_searcher(search_str:str, search_list:str):

    search_obj = re.findall(search_list, search_str)
    if search_obj :
        #return_str = search_str[search_obj.start(): search_obj.end()]
        return_str =', '.join(search_obj)
    else:
        return_str = 'NA'
    return return_str

new_df['Noun Phrase'] = new_df['SENTENCE'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern))


# # SPLITTING PHRASES BASED ON PUNCTUATION

# In[ ]:


# Separating findings based on ','

s = new_df["Noun Phrase"].str.split(',').apply(pd.Series,1).stack()
s.index = s.index.droplevel(-1) # to line up with df's index
s.name = 'Comment_Text' # needs a name to join

# There are blank or emplty cell values after above process. Removing them
s.replace('', np.nan, inplace=True)
s.replace(' ', np.nan, inplace=True)
s.dropna(inplace=True)

df3 = new_df.join(s.str.strip())


# In[ ]:


# Separating noun and adjtives
df3['PhrasePOSTags'] = nltk.pos_tag_sents(df3['Comment_Text'].apply(word_tokenize).tolist())

# Extracting Noun
def ExtractNoun(text):
    only_tagged_nouns = []
    for word, pos in text:
        if (pos == 'NN' or pos == 'NNPS' or pos=='NNS' or pos=='NNP'):
            only_tagged_nouns.append(word)
        into_string=str(only_tagged_nouns)
    return(into_string)
    
df3['Noun'] = df3['PhrasePOSTags'].apply(lambda x: ExtractNoun(x))
df3['Noun'] = df3.Noun.apply(lambda x: str(x).strip('[]'))
df3['Noun'] = df3['Noun'].str.replace("'", "")


# # EXTRACTING ADJECTIVES AND ADVERBS USING RULE BASED METHOD TO IDENTIFY SENTIMENTS ASSOCIATED WITH THE NOUN PHRASES OR ASPECTS

# In[ ]:


# Extracting Adjectives
def ExtractAdj(text):
    only_tagged_adj = []
    for word, pos in text:
        if (pos == 'JJ' or pos=='JJR' or pos=='JJS' or pos=='VBG'):
            only_tagged_adj.append(word)
    return(only_tagged_adj)
    
df3['Adjectives'] = df3['PhrasePOSTags'].apply(lambda x: ExtractAdj(x))
df3['Adjectives'] = df3.Adjectives.apply(lambda x: str(x).strip('[]'))
df3['Adjectives'] = df3['Adjectives'].str.replace("'", "")


# In[ ]:


# Extracting adverbs
def ExtractAdv(text):
    only_tagged_nouns = []
    for word, pos in text:
        if (pos == 'RB' or pos=='RBS' or pos=='RBR'):
            only_tagged_nouns.append(word)
    return(only_tagged_nouns)

df3['Adverbs'] = df3['PhrasePOSTags'].apply(lambda x: ExtractAdv(x))
df3['Adverbs'] = df3.Adverbs.apply(lambda x: str(x).strip('[]'))
df3['Adverbs'] = df3['Adverbs'].str.replace("'", "")


# In[ ]:


df3.to_csv(outpath+"DF-NNJJVB-2019-23.csv")

df3=pd.read_csv(outpath+'DF-NNJJVB-2019-23.csv')



# # COMPUTING FREQUENCY OF ADJECTIVES RELATED TO HOST

# In[ ]:


df3.drop(columns={'SentenceLenght','POSTags','Noun Phrase','Comment_Text','PhrasePOSTags','Adverbs'}, inplace= True)# 'Unnamed: 0',

# Creating a boolean mask/filter to check for the presence of host
Mask= df3['Noun']=='host'
HostDF=df3[Mask]
HostDF.dtypes

HostAdjDistribution=HostDF.groupby(['Adjectives']).Adjectives.count().rename("Frequency").reset_index()

# Reading Dictionary word + category
WordDic=pd.read_excel(path +'WordList-2.xls', sheet_name='AllCategories')
WordDic.dtypes
WordDic['Word']=WordDic['Word'].str.lower()

WordDic.rename(columns={'Word':'Adjectives'}, inplace= True)            

HostDF= pd.merge(HostDF,WordDic, how= 'left', on='Adjectives')    


# In[ ]:


# Checking null in Category column 
CheckDF= HostDF[HostDF['Category'].isnull()]

CheckDFd=CheckDF.groupby(['Adjectives']).Adjectives.count().rename("Frequency").reset_index()
CheckDFd.to_csv(outpath+'tobeCheckedAdj.csv')

# Applying pd_pivot on Category column,to get column and the value as either 0 or 1
HostDF.dtypes
Test= pd.pivot_table(HostDF,index=['ID','Reviewer_ID','SENTENCE','Noun'], columns='Category', aggfunc='count', fill_value=0)

# Fixing the double header and reseting the index
Test.columns = Test.columns.droplevel(0)
Test.reset_index(inplace=True)
Test.dtypes

# Aggregating adjectives to comment level
HostAdjDF = Test.groupby(['ID', 'Reviewer_ID'])[['Arousal', 'ComForm', 'Emot', 'Eval', 'Feel', 'Pain', 'Pleasure', 'Vice', 'Virtue', 'affOth', 'affTot', 'Think', 'Compare']].sum().reset_index()

HostAdjDF.rename(columns={'Arousal':'HostArousal','ComForm':'HostComForm','Emot':'HostEmot','Eval':'HostEval','Feel':'HostFeel','Think':'HostThink','Compare':'HostCompare','Pain':'HostPain','Pleasure':'HostPleasure','Vice':'HostVice','Virtue':'HostVirtue','affOth':'HostOffOth','affTot':'HostOffTot'}, inplace= True)#
HostAdj_Describe=HostAdjDF.describe().transpose()

HostAdjDF.to_csv(outpath+'HostAdjDF.csv')
HostAdj_Describe.to_csv(outpath+'HostAdj_Describe.csv')


# In[ ]:


HostAdjDF


# # ASSIGNING CATEGORY TO AMENITIES ADJECTIVES

# In[ ]:


# Assigning Category to amenities adjectives 
mylist1=pd.read_excel(path +'WordList-2.xls', sheet_name='Amenities')
Amenities=mylist1.iloc[:,0]
pattern = '|'.join(Amenities)

df3['Noun']=df3['Noun'].apply(str)
df3['Amenities'] = df3['Noun'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern))

df3['Amenities']=df3['Amenities'].apply(str)

AmenitiesDF=df3[df3['Amenities'] != "NA" ]
LocationAdjDistribution=AmenitiesDF.groupby(['Adjectives']).Adjectives.count().rename("Frequency").reset_index()

AmenitiesDF= pd.merge(AmenitiesDF,WordDic, how= 'left', on='Adjectives')    
AmenitiesDF.dtypes
CheckDF= AmenitiesDF[AmenitiesDF['Category'].isnull()]

CheckDFd=CheckDF.groupby(['Adjectives']).Adjectives.count().rename("Frequency").reset_index()
CheckDFd.to_csv(outpath+'tobeCheckedAdjLoc.csv')


# In[ ]:


AmenitiesDF


# In[ ]:


# Applying pivot on Category column, to get column and the value in either 0 or 1 indicating presence or absence
Test= pd.pivot_table(AmenitiesDF,index=['ID','Reviewer_ID','SENTENCE','Noun','Amenities'], columns='Category', aggfunc='count', fill_value=0)

# Sorting out double header and reset the index
Test.columns = Test.columns.droplevel(0)
Test.reset_index(inplace=True)
Test.dtypes

# Aggregating adjectives to te comment level

AmenitiesAdjDF = Test.groupby(['ID', 'Reviewer_ID'])[['Arousal', 'ComForm', 'Emot', 'Eval', 'Pain', 'Pleasure', 'Vice', 'Virtue', 'affOth', 'affTot', 'Compare', 'Think']].sum().reset_index()
AmenitiesAdjDF.rename(columns={'Compare':'AmenitiesCompare','Think':'AmenitiesThink','Arousal':'AmenitiesArousal','ComForm':'AmenitiesComForm','Emot':'AmenitiesEmot','Eval':'AmenitiesEval','Feel':'AmenitiesFeel','Pain':'AmenitiesPain','Pleasure':'AmenitiesPleasure','Vice':'AmenitiesVice','Virtue':'AmenitiesVirtue','affOth':'AmenitiesOffOth','affTot':'AmenitiesOffTot'}, inplace= True)#'Positive':'LocationPositive','Negative':'LocationNegative',
AmenitiesAdjDF_Describe=AmenitiesAdjDF.describe().transpose()

AmenitiesAdjDF.to_csv(outpath+'AmenitiesAdjDF.csv')
AmenitiesAdjDF_Describe.to_csv(outpath+'AmenitiesAdj_Describe.csv')


# # FINDING AMENITIES MOSTLY MENTIONED BY RETURNING AND NON RETURNING CUSTOMERS

# In[ ]:


# Creating a Returner Flag
df['Returner'] = df.groupby(['listing_id', 'reviewer_id'])['reviewer_id'].transform('count') > 1
df['Returner'] = df['Returner'].astype(int)

# Defining certain Amenity Keywords
amenities_keywords = {
    'Wi-Fi': ['wifi', 'internet', 'broadband'],
    'Comfortable Bed': ['comfortable bed', 'bed', 'mattress'],
    'Kitchen': ['kitchen', 'stove', 'oven', 'cookware'],
    'Parking': ['parking', 'car park', 'garage'],
    'Bathroom': ['bathroom', 'shower', 'toilet'],
}

# Searching for Amenity Mentions
for amenity, keywords in amenities_keywords.items():
    pattern = '|'.join(keywords)
    df[amenity] = df['comments'].str.contains(pattern, case=False, regex=True).astype(int)

# Calculating Frequency for Returning and Non-Returning Guests
returner_amenity_frequency = df[df['Returner'] == 1][list(amenities_keywords.keys())].mean()
non_returner_amenity_frequency = df[df['Returner'] == 0][list(amenities_keywords.keys())].mean()

#  Comparison
comparison = pd.DataFrame({
    'Returner Frequency': returner_amenity_frequency,
    'Non-Returner Frequency': non_returner_amenity_frequency
})
comparison['Difference'] = comparison['Returner Frequency'] - comparison['Non-Returner Frequency']

#  Visualising the difference in frequency of mentions of amenities
comparison.plot(kind='bar', figsize=(10, 6), title='Amenity Mention Frequency: Returning vs. Non-Returning Guests')
print(comparison)



# In[ ]:


# Plotting the comparison between returners and non returners
plt.figure(figsize=(10, 6))
comparison[['Returner Frequency', 'Non-Returner Frequency']].plot(kind='bar', figsize=(12, 8), color=['skyblue', 'lightcoral'])

# Customizing the plot
plt.title('Amenity Mention Frequency: Returning vs. Non-Returning Guests', fontsize=16)
plt.xlabel('Amenities', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(['Returning Guests', 'Non-Returning Guests'], fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[ ]:


# Storing reviews by returners
returning_reviews = df[df['Returner'] == 1]['comments']

# Defining certain Aspect Keywords for analysis
aspect_keywords = {
    'Amenities': ['wifi', 'kitchen', 'parking', 'bed', 'bathroom', 'amenities', 'shower', 'toilet', 'clean', 'space', 'spacious', 'comfortable', 'facilities'],
    'Environment': ['location', 'view', 'neighborhood', 'area', 'environment', 'surroundings', 'quiet', 'garden', 'park', 'central', 'scenic'],
    'Worth': ['value', 'price', 'cost', 'worth', 'expensive', 'cheap', 'affordable'],
    'Outcome': ['experience', 'satisfaction', 'enjoyment', 'happy', 'pleased', 'content', 'wonderful'],
    'Recommend': ['recommend', 'suggest', 'advise', 'recommendation', 'referral']
}

# Counting Aspect Mentions
def count_aspects(text, aspect_keywords):
    aspect_count = {aspect: 0 for aspect in aspect_keywords}
    words = word_tokenize(text.lower())
    for word in words:
        for aspect, keywords in aspect_keywords.items():
            if word in keywords:
                aspect_count[aspect] += 1
    return aspect_count

# Applying count_aspects function to each returning review
aspect_counts = returning_reviews.apply(lambda x: count_aspects(x, aspect_keywords))

# Aggregating the counts across all returning guests
total_aspect_counts = Counter()
for aspect_count in aspect_counts:
    total_aspect_counts.update(aspect_count)

# Converting to DataFrame for easier visualization
aspect_df = pd.DataFrame.from_dict(total_aspect_counts, orient='index', columns=['Frequency']).reset_index()
aspect_df.rename(columns={'index': 'Aspect'}, inplace=True)

# Visualising the Results
import matplotlib.pyplot as plt

aspect_df.plot(kind='bar', x='Aspect', y='Frequency', legend=False, color='skyblue')
plt.title('Frequency of Aspects Mentioned by Returning Guests')
plt.xlabel('Aspect')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# In[70]:


import pandas as pd
import nltk
import importlib.metadata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
import seaborn as sns
import numpy as np
import wordcloud

# Print the versions of packages
print("pandas:", pd.__version__)
print("nltk:", nltk.__version__)
print("langdetect:", importlib.metadata.version('langdetect'))
print("unidecode:", importlib.metadata.version('Unidecode'))
print("vaderSentiment:", importlib.metadata.version('vaderSentiment'))
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("numpy:", np.__version__)
print("wordcloud:", wordcloud.__version__)


# In[ ]:




