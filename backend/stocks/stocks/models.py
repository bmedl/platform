from django.db import models
from django.contrib.auth.models import User
from django.utils.encoding import python_2_unicode_compatible

#https://developer.oanda.com/rest-live-v20/pricing-df/
@python_2_unicode_compatible
class Stock_USDEUR(models.Model):
    types = models.CharField(max_length=120)
    event_date = models.DateTimeField('Event Date')
    bid = models.DecimalField(max_digits=15, decimal_places=10)
    ask = models.DecimalField(max_digits=15, decimal_places=10)
    closeoutBid = models.DecimalField(max_digits=15, decimal_places=10)
    closeoutAsk = models.DecimalField(max_digits=15, decimal_places=10)