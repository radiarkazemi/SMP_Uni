from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


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


class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea)


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
