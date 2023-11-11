from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import requests
import pandas as pd
from data_sci.models import MovieDescription
from data_sci.models import MovieReview
from data_sci.models import MovieReviewRelated
from data_sci.models import Comment
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pandas as pd
from data_sci.models import MovieDescription, MovieReview, MovieReviewRelated
from io import StringIO
import csv

import requests
import csv
from io import StringIO
from django.http import JsonResponse

import requests
import csv
from io import StringIO
from django.http import JsonResponse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Add this import



def add_comment(request):
    if request.method == "POST":
        try:
            user = request.user
            comment_text = request.POST.get("comment_text")

            if not comment_text:
                return JsonResponse({'error': 'Comment cannot be empty.'}, status=400)

            comment = Comment(user=user, text=comment_text)
            comment.save()

            return JsonResponse({'success': 'Comment added successfully'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


def Visualize_D3(request):
    # Fetch data from the MovieReview model
    reviews = MovieReview.objects.all()

    total_Positivereview = [review.imbd_rating for review in reviews]
    total_Negativereview = [review.imbd_rating for review in reviews]

    X = np.array(total_Positivereview).reshape(-1, 1)
    y = np.array(total_Negativereview).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    regr = LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X)

    # Fetch titles in the same order as the data
    titles = [review.review_title for review in reviews]

    # Create a list of dictionaries for the output
    json_output = []
    for i in range(len(titles)):
        json_output.append({
            'title': titles[i],
            'total_Positive': total_Positivereview[i],
            'total_Negative': total_Negativereview[i],
            'predict_applicants': float(y_pred[i][0])
        })

    # Return the data as a JSON response
    return JsonResponse(json_output, safe=False)



def external_api_get_from_sheet(request):
    # Fetching the CSV data from the Google Sheets link
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv"
    sheet_response = requests.get(sheet_url)
    sheet_response.raise_for_status()  # Will raise an exception if the request is not successful

    # Parsing the CSV data
    csvfile = StringIO(sheet_response.text)
    reader = csv.reader(csvfile)
    header = next(reader)  # Assuming the first row is the header

    # Let's say we want the text from the second row (index 1) and the column named 'review'
    review_index = header.index('review')
    next(reader)  # Skip the second row
    row = next(reader)  # Get the third row
    review_text = row[review_index]

    # Now we have the review text, we can make the API call
    url = "https://api.meaningcloud.com/sentiment-2.1"
    params = {
        'key': "aeefdec613582d7f59ae105975ed2103",
        'txt': review_text
    }
    response = requests.get(url, params=params)

    # Check the response status code
    if response.status_code == 200:
        try:
            result = response.json()
        except ValueError:
            # Handle the exception if the response is not in JSON format
            result = {'error': 'Invalid JSON response', 'content': response.text}
    else:
        result = {'error': 'API request failed', 'status_code': response.status_code}

    return JsonResponse(result)



def analyze_sentiment(request):
    if request.method == 'GET':
        # Handle GET request here
        text = request.GET.get('text')  # Get the text from the query parameters
        
        if text:
            result = perform_sentiment_analysis(text)
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Text parameter is missing in the GET request.'})

    elif request.method == 'POST':
        text = request.POST.get('text')  # Get the text from the POST request
        
        if text:
            result = perform_sentiment_analysis(text)
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Text parameter is missing in the POST request.'})

    else:
        return JsonResponse({'error': 'This endpoint only supports GET and POST requests.'})
    

    
def load_imdb_data(request):
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv'
    imdb_data = pd.read_csv(url)
    return JsonResponse(imdb_data.head(10).to_json(orient="records"), safe=False)

def imdb_data_summary(request):
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv'
    imdb_data = pd.read_csv(url)
    
    # Using .describe() to get the summary statistics of the dataframe
    data_summary = imdb_data.describe()
    
    # Converting the summary dataframe to JSON format
    return JsonResponse(data_summary.to_json(orient="index"), safe=False)

def imdb_sentiment_count(request):
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv'
    imdb_data = pd.read_csv(url)
    
    # Counting the occurrences of each unique value in the 'sentiment' column
    sentiment_counts = imdb_data['sentiment'].value_counts()
    
    # Converting the counts to JSON format
    return JsonResponse(sentiment_counts.to_json(), safe=False)


def index(request):
    context = {
        "title": "Django example",
    }
    return render(request, "index.html", context)

def homepage(request):
    context={}
    return render(request,"data_sci/innovitech/index.html",context=context)

def blog(request):
    context={}
    return render(request,"data_sci/innovitech/Blog.html",context=context)

def timeline(request):
    context={}
    return render(request,"data_sci/innovitech/timeline.html",context=context)

def review(request):
    context={}
    return render(request,"data_sci/innovitech/review.html",context=context)

def import_data_csv(request):
    csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTP9SJX5VtWy6eQy7QZRrA2cuyqucI2-f-sZht2vQzRkbOBvmJlCQYZyg1F99U7Xcsk8EpOwgGZmFNG/pub?output=csv"
    df = pd.read_csv(csv_url)
    data_sets = df[["movie_id","title","year","genre","storyline","user_id",
                    "user_name","review_id","review_title","review_content","imbd_rating"]]
    success = []
    errors = []
    for index, row in data_sets.iterrows():
        instance = MovieDescription(
            movie_id = row['movie_id'],
            title = row['title'],
            year = int(row["year"]) if pd.notnull(row["year"]) and str(row["year"]).isdigit() else None,
            genre = row['genre'],
            storyline = row['storyline'],
        )

        movie_review_instance = MovieReview(
            user_id=row['user_id'],
            user_name=row['user_name'],
            review_id=row['review_id'],
            review_title=row['review_title'],
            review_content=row['review_content'],
            imbd_rating=int(row['imbd_rating'])
            )
        movie_review_related_instance = MovieReviewRelated(
            movie_description=instance,
            movie_review=movie_review_instance
            )
    
        try:
            instance.save()
            
            movie_review_instance.save()  
            movie_review_related_instance.save()  
            success.append(index)
        except:
            errors.append(index)
    return JsonResponse({"success_indices": success, "error_indices": errors})

