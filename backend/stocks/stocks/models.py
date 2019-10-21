from django.db import models
from django.contrib.auth.models import User
from django.utils.encoding import python_2_unicode_compatible

#https://developer.oanda.com/rest-live-v20/pricing-df/
@python_2_unicode_compatible
class Stock(models.Model):
    name = models.CharField(max_length=120)
    event_date = models.DateTimeField(auto_now=True)
    price_date = models.DateTimeField()
    bid = models.DecimalField(max_digits=15, decimal_places=10)
    liquidity = models.IntegerField()
    ask = models.DecimalField(max_digits=15, decimal_places=10)
    closeout_bid = models.DecimalField(max_digits=15, decimal_places=10)
    closeout_ask = models.DecimalField(max_digits=15, decimal_places=10)
    tradeable = models.BooleanField()
