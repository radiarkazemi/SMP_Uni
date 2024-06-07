from django import forms


class StockTickerForm(forms.Form):
    ticker = forms.ChoiceField(
        choices=[('', '__select one__')],  # Add a default placeholder option
        label='Stock Ticker',
        widget=forms.Select(
            attrs={'class': 'form-control mr-2', 'onchange': 'submitForm();'})
    )

    def __init__(self, *args, **kwargs):
        stock_choices = kwargs.pop('stock_choices', [])
        super().__init__(*args, **kwargs)
        # Append the stock choices
        self.fields['ticker'].choices += stock_choices
