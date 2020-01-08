from rest_framework.viewsets import ModelViewSet
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


@api_view(['POST'])
def predict(request):
    from ..neuralnet.run import predict as predict_fn
    try:
        data = request.data
        predict_fn(data['name'])
        return Response({'success': True})
    except e:
        return Response({'success': False, 'error': e})
