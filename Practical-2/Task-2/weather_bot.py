import telebot
import requests

BOT_TOKEN = 'token in report'
WEATHER_API_KEY = 'token in report'

bot = telebot.TeleBot(BOT_TOKEN)

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=ua"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
    if response.status_code == 200:
        try:
            data = response.json()
            weather = {
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed']
            }
            return weather
        except KeyError:
            print("Error: Invalid data format returned by OpenWeatherMap API")
            return None
    else:
        print(f"Error: OpenWeatherMap API returned status code {response.status_code}")
        return None

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, "Привіт! Введіть назву міста, щоб отримати прогноз погоди.")

@bot.message_handler(content_types=['text'])
def send_weather(message):
    city = message.text.strip()
    weather = get_weather(city)
    if weather:
        response = (
            f"Погода в {city}:\n"
            f"Температура: {weather['temperature']}°C\n"
            f"Опис: {weather['description']}\n"
            f"Швидкість вітру: {weather['wind_speed']} м/с"
        )
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "Не вдалося знайти інформацію про погоду для цього міста. Перевірте назву міста та спробуйте знову.")

@bot.message_handler(commands=['menu'])
def menu(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button = telebot.types.KeyboardButton("Отримати погоду")
    markup.add(button)
    bot.send_message(message.chat.id, "Оберіть дію:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Отримати погоду")
def ask_city(message):
    bot.reply_to(message, "Введіть назву міста:")

bot.polling(none_stop=True)
