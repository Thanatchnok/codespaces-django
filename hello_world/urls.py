"""hello_world URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from hello_world.core import views as core_views
from data_sci.models import *


urlpatterns = [
    path("", core_views.homepage, name='homepage'),
    path("admin/", admin.site.urls),
    path("__reload__/", include("django_browser_reload.urls")),
    path("example/blog", core_views.blog, name='blog'),
    path("example/timeline", core_views.timeline, name='timeline'),
    path("example/review", core_views.review, name='review'),
    path("example/importreview", core_views.import_datareview_csv),
    path("example/import", core_views.import_data_csv),
    path("example/tdata", core_views.load_imdb_data),
    path("example/sdata", core_views.imdb_data_summary),
    path("example/cdata", core_views.imdb_sentiment_count),
    path("example/getAPI", core_views.external_api_get_from_sheet),
    path("example/visualize_d3", core_views.visualize_d3),
    path("example/comment",core_views.add_comment),
    path('example/visualize_d3/', core_views.visualize_d3, name='visualize_d3'),
    
    ]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
