from rest_framework.viewsets import ModelViewSet, ViewSet
from .serializers import StockSerializer
from .models import Stock
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.decorators import action, api_view


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


class Predict(ViewSet):
    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        import os
        from os import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

        from stocks.neuralnet.run import predict as predict_fn

        try:
            data = request.data
            predict_fn(data['name'])
            return Response({'success': True})
        except Exception as e:
            if hasattr(e, 'message'):
                return Response({'success': False, 'error': e.message})
            else:
                return Response({'success': False, 'error': str(e)})

