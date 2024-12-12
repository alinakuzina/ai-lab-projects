import telebot
from currency_converter import CurrencyConverter

BOT_TOKEN = 'token in report'

bot = telebot.TeleBot(BOT_TOKEN)
currency = CurrencyConverter()

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, "Привіт! Введіть суму і валюти у форматі '100 USD to EUR' для конвертації.")

@bot.message_handler(content_types=['text'])
def convert_currency(message):
    try:
        text = message.text.strip()
        if " to " in text:
            parts = text.split(" to ")
            amount, from_currency = parts[0].split()
            to_currency = parts[1]

            amount = float(amount)
            result = currency.convert(amount, from_currency.upper(), to_currency.upper())

            bot.reply_to(message, f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}")
        else:
            bot.reply_to(message, "Будь ласка, використовуйте формат '100 USD to EUR'")
    except ValueError:
        bot.reply_to(message, "Неправильний формат. Перевірте введені дані та спробуйте ще раз.")
    except Exception as e:
        bot.reply_to(message, f"Сталася помилка: {str(e)}. Перевірте введені дані.")

bot.polling(none_stop=True)
