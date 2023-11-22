from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Review(models.Model):
    review = models.CharField(max_length=20)
    sentiment = models.CharField(max_length=20)

def __str__(self):
    return f"{self.review.capitalize()} - {self.sentiment}"

#Not use (use just for create web app)
class MovieDescription(models.Model):
    
    movie_id = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    year = models.IntegerField()
    genre = models.CharField(max_length=255)
    storyline = models.CharField(max_length=255)
    

class MovieReview(models.Model):

    user_id = models.CharField(max_length=255)
    user_name = models.CharField(max_length=255)
    review_id = models.CharField(max_length=255)
    review_title = models.CharField(max_length=255)
    review_text = models.TextField()
    imbd_rating = models.FloatField(default=0.0, null=True)
    sentiment = models.CharField(max_length=20)

class Comment(models.Model):
    text = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

def __str__(self):
    return f"Comment #{self.id}"
    

class MovieReviewRelated(models.Model):

    movie_description = models.ForeignKey(MovieDescription, on_delete=models.CASCADE)
    movie_review = models.ForeignKey(MovieReview, on_delete=models.CASCADE)

def __str__(self):
    return f"{self.movie_description.title} - {self.movie_review.review_title}"

def __str__(self):
        return f"Related Review: {self.movie_review.review_title}"





