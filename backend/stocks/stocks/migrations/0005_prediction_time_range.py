# Generated by Django 2.2.7 on 2019-12-15 22:00

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stocks', '0004_auto_20191212_1533'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='time_range',
            field=models.DurationField(default=datetime.timedelta(0)),
            preserve_default=False,
        ),
    ]