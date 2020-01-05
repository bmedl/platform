from django.db import models
from django.contrib.auth.models import User
from django.utils.encoding import python_2_unicode_compatible


# https://developer.oanda.com/rest-live-v20/pricing-df/
class Stock(models.Model):
    """
    Various data describing a stock.
    """
    name = models.CharField(max_length=120)
    event_date = models.DateTimeField(auto_now=True)
    price_date = models.DateTimeField()
    bid = models.DecimalField(max_digits=15, decimal_places=10)
    bid_liquidity = models.IntegerField()
    ask = models.DecimalField(max_digits=15, decimal_places=10)
    ask_liquidity = models.IntegerField()
    closeout_bid = models.DecimalField(max_digits=15, decimal_places=10)
    closeout_ask = models.DecimalField(max_digits=15, decimal_places=10)
    tradeable = models.BooleanField()


class NetworkModel(models.Model):
    """
    A trained Keras model saved as a blob in hdf5 format.
    """
    name = models.CharField(max_length=120, blank=True, default='')
    updated = models.DateTimeField(auto_now=True)
    model_blob = models.BinaryField()


class Prediction(models.Model):
    """
    A predicted value.
    """
    created = models.DateTimeField(auto_now=True)
    time_range = models.DurationField()
    price_date = models.DateTimeField()
    name = models.CharField(max_length=120)
    value = models.IntegerField()

class BacktestResult(models.Model):
    """
    A predicted value.
    """
    created = models.DateTimeField(auto_now=True)
    price_date = models.DateTimeField()
    expected = models.IntegerField()
    actual = models.IntegerField()
