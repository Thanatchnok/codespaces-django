# Generated by Django 4.1.10 on 2023-11-03 12:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("data_sci", "0003_alter_moviedescription_rank"),
    ]

    operations = [
        migrations.AlterField(
            model_name="moviedescription",
            name="rank",
            field=models.IntegerField(),
        ),
    ]
