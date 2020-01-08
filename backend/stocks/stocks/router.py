from .restapi.viewsets import ListViewSet, predict
from rest_framework import routers

router = routers.DefaultRouter(trailing_slash=False)
router.register('stocks', ListViewSet, base_name='stocks')
router.register('predict', predict, base_name='predict')