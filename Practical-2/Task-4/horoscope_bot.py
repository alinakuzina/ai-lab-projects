import telebot
import requests

BOT_TOKEN = 'token in report'
RAPIDAPI_KEY = 'token in report'

bot = telebot.TeleBot(BOT_TOKEN)

zodiac_signs = {
    "Aries": "aries", "Taurus": "taurus", "Gemini": "gemini",
    "Cancer": "cancer", "Leo": "leo", "Virgo": "virgo",
    "Libra": "libra", "Scorpio": "scorpio", "Sagittarius": "sagittarius",
    "Capricorn": "capricorn", "Aquarius": "aquarius", "Pisces": "pisces"
}

@bot.message_handler(commands=['start'])
def start_message(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for sign in zodiac_signs:
        markup.add(telebot.types.KeyboardButton(sign))
    bot.send_message(
        message.chat.id,
        "Hello! Choose your zodiac sign to get your horoscope:",
        reply_markup=markup
    )

@bot.message_handler(content_types=['text'])
def horoscope_handler(message):
    sign = message.text.strip()
    if sign in zodiac_signs:
        try:
            english_sign = zodiac_signs[sign]
            horoscope = get_horoscope(english_sign)
            bot.reply_to(message, f"Your horoscope ({sign}):\n{horoscope}")
        except Exception as e:
            bot.reply_to(message, f"An error occurred while fetching the horoscope: {str(e)}")
    else:
        bot.reply_to(message, "Please choose a valid zodiac sign.")

def get_horoscope(sign):
    url = f"https://best-daily-astrology-and-horoscope-api.p.rapidapi.com/api/Detailed-Horoscope/"
    headers = {
        "x-rapidapi-host": "best-daily-astrology-and-horoscope-api.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    querystring = {"zodiacSign": sign}

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        if "prediction" in data:
            return data["prediction"]
        else:
            raise Exception("Horoscope not found for this sign.")
    else:
        raise Exception(f"API access error. Status code: {response.status_code}")

bot.polling(none_stop=True)
