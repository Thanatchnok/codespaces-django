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
from data_sci.models import Review
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pandas as pd

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



from django.db.models import Count
from django.http import JsonResponse
# views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from django.http import JsonResponse
import json
from django.shortcuts import render
from django.http import JsonResponse


@csrf_exempt
def add_comment(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            comment_text = data.get('comment_text')

            if comment_text:
                # Save the comment in your Comment model
                new_comment = Comment.objects.create(text=comment_text)
                return JsonResponse({
                    'id': new_comment.id,
                    'comment_text': new_comment.text,
                    'timestamp': new_comment.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                return JsonResponse({'error': 'No comment provided'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

from django.views.decorators.csrf import csrf_exempt


from django.db.models import Count
def visualize_d3(request):
    reviews = MovieReview.objects.filter(sentiment__in=['Positive', 'Negative'])
    sentiment_counts = reviews.values('sentiment').annotate(count=Count('sentiment'))

    # Prepare data for the bar graph
    labels = [entry['sentiment'] for entry in sentiment_counts]
    counts = [entry['count'] for entry in sentiment_counts]

    if 'Positive' not in labels:
        labels.append('Positive')
        counts.append(0)
    if 'Negative' not in labels:
        labels.append('Negative')
        counts.append(0)

    json_output = {
        'sentiment': labels,
        'counts': counts,
    }

    return JsonResponse(json_output)

def external_api_get_from_sheet(request):
   
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv"
    sheet_response = requests.get(sheet_url)
    sheet_response.raise_for_status()  

    # Parsing the CSV data
    csvfile = StringIO(sheet_response.text)
    reader = csv.reader(csvfile)
    header = next(reader)  

   
    review_index = header.index('review')
    next(reader)
    row = next(reader)  
    review_text = row[review_index]

    
    url = "https://api.meaningcloud.com/sentiment-2.1"
    params = {
        'key': "c07e09e9aef8ace95ebd7370dfc90cce",
        'txt': review_text
    }
    response = requests.get(url, params=params)

   
    if response.status_code == 200:
        try:
            result = response.json()

           
            sentiment_data = result.get('sentence_list', [])

            
            agreement_data = []
            score_tags = []

            for sentence in sentiment_data:
                agreement_info = sentence.get('agreement', '')
                score_tag = sentence.get('score_tag', '')
                
                if agreement_info:
                    agreement_data.append({'agreement': agreement_info})
                
                if score_tag:
                    score_tags.append(score_tag)

            num_positive = sum(tag == 'P' or tag == 'P+' for tag in score_tags)
            num_negative = sum(tag == 'N' or tag == 'N+' for tag in score_tags)
            num_neutral = sum(tag == 'NEU' or tag == 'NONE' for tag in score_tags)

            if num_neutral > num_positive:
                overall_sentiment = 'Neutral'
            elif num_negative > 2:
                overall_sentiment = 'Negative'
            else:
                overall_sentiment = 'Positive'


           
            soup = BeautifulSoup(review_text, 'html.parser')
            cleaned_review_text = soup.get_text()

           
            response_data = {
                'review_text': cleaned_review_text,
                'agreement_data': agreement_data,
                'score_tags': score_tags,
                'overall_sentiment': overall_sentiment,
            }

           
            if overall_sentiment not in ['Positive', 'Negative']:
                response_data['error'] = 'No Positive or Negative sentiment found'

            
            movie_review_instance = MovieReview.objects.create(
                review_text=cleaned_review_text,
                sentiment=overall_sentiment,
            )

            return JsonResponse(response_data)

        except ValueError:
           
            result = {'error': 'Invalid JSON response', 'content': response.text}
    else:
        result = {'error': 'API request failed', 'status_code': response.status_code}

    return JsonResponse(result)

    
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


#This is csv of review and sentiment of imbd
def import_datareview_csv(request):
    csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSWaIu5uI0oQ6tCXZBsdT6lqgXvbjbzg3ZXJnK2krqVmNC33fn62Fd0mPA3EXMzv5E9eweW6ZDWvQO4/pub?output=csv"
    df = pd.read_csv(csv_url)
    
    print("Columns in DataFrame:", df.columns)

    data_sets = df[["review", "sentiment"]]
    success = []
    errors = []

    for index, row in data_sets.iterrows():
        instance = Review(
            review=row['review'],
            sentiment=row['sentiment'],
        )

        try:
            instance.save()
            success.append(index)
        except Exception as e:
            errors.append({"index": index, "error": str(e)})

    return JsonResponse({"success_indices": success, "error_indices": errors})


#not use
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

