from rest_framework.viewsets import ModelViewSet
from .serializers import StockSerializer
from .models import Stock
from rest_framework.response import Response
from rest_framework.decorators import action

class ListViewSet(ModelViewSet):
    queryset = Stock.objects.all()
    serializer_class = StockSerializer

class Last100Stocks(ModelViewSet):
    queryset = Stock.objects.all()
    serializer_class = StockSerializer

    @action(methods=['get'], detail=False)
    def newest(self, request):
        newest = self.get_queryset().last()
        serializer = self.get_serializer_class()(newest)
        return Response(serializer.data)