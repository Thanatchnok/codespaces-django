# Generated by Django 4.1.10 on 2023-11-03 12:12

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("data_sci", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="moviereview",
            old_name="imdb_rating",
            new_name="imbd_rating",
        ),
    ]
