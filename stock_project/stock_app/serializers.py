from rest_framework import serializers
import datetime


class DateTimeToDateField(serializers.DateField):
    def to_representation(self, value):
        if isinstance(value, datetime.datetime):
            return value.date()
        return super().to_representation(value)


class StockDataUniSerializer(serializers.Serializer):
    Date = DateTimeToDateField()
    Open = serializers.FloatField(allow_null=True)
    High = serializers.FloatField(allow_null=True)
    Low = serializers.FloatField(allow_null=True)
    Close = serializers.FloatField(allow_null=True)
    Volume = serializers.IntegerField(allow_null=True)


class StockDataPredSerializer(serializers.Serializer):
    Date = DateTimeToDateField()
    Time = serializers.IntegerField()
    jalali_date = serializers.CharField()
    gregorian_date = serializers.CharField()
    stock_value = serializers.FloatField()
