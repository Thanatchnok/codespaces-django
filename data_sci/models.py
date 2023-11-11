from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


# Create your models here.
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
    review_content = models.CharField(max_length=255)
    imbd_rating = models.IntegerField()
    

class MovieReviewRelated(models.Model):

    movie_description = models.ForeignKey(MovieDescription, on_delete=models.CASCADE)
    movie_review = models.ForeignKey(MovieReview, on_delete=models.CASCADE)
 


def __str__(self):
    return f"{self.movie_description.title} - {self.movie_review.review_title}"


# models.py
from django.db import models

class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.text}'

