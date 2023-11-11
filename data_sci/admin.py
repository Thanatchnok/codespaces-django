from django.contrib import admin
from data_sci.models import *
# Register your models here.

@admin.register(MovieDescription)
class MovieDescriptionAdmin(admin.ModelAdmin):
    pass

@admin.register(MovieReview)
class MovieReviewAdmin(admin.ModelAdmin):
    pass

@admin.register(MovieReviewRelated)
class MovieReviewRelatedAdmin(admin.ModelAdmin):
    pass
