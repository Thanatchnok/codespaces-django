# Generated by Django 4.1.10 on 2023-11-17 11:15

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("data_sci", "0011_alter_moviereview_sentiment"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="moviereview",
            name="review_content",
        ),
        migrations.AddField(
            model_name="moviereview",
            name="review_text",
            field=models.TextField(default="Positive"),
            preserve_default=False,
        ),
    ]