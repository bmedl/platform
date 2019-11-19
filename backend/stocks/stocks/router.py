from .restapi.viewsets import ListViewSet
from rest_framework import routers

router = routers.DefaultRouter()
router.register('stocks', ListViewSet, base_name='stocks')