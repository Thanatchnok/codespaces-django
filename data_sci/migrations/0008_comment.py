# Generated by Django 4.1.10 on 2023-11-07 10:43

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("data_sci", "0007_alter_moviereview_review_id_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="Comment",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
