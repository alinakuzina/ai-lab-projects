import os
import telebot
import sqlite3

BOT_TOKEN = 'token in report'

os.makedirs('db', exist_ok=True)

def initialize_database():
    db_path = 'db/roshen.db'

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('SELECT 1')
            conn.close()
        except sqlite3.DatabaseError:
            os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL
        )
    ''')
    cursor.execute('SELECT COUNT(*) FROM products')
    if cursor.fetchone()[0] == 0:
        products = [
            ("Шоколад Roshen", 50.0),
            ("Цукерки Roshen", 120.0),
            ("Торт Roshen", 350.0),
        ]
        cursor.executemany('INSERT INTO products (name, price) VALUES (?, ?)', products)
        conn.commit()
    conn.close()

initialize_database()

bot = telebot.TeleBot(BOT_TOKEN)

def get_products():
    conn = sqlite3.connect('db/roshen.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, price FROM products')
    products = cursor.fetchall()
    conn.close()
    return products

def show_menu(chat_id):
    products = get_products()
    if not products:
        bot.send_message(chat_id, "Наразі немає доступних товарів.")
        return
    markup = telebot.types.InlineKeyboardMarkup()
    for product in products:
        markup.add(telebot.types.InlineKeyboardButton(
            text=f"{product[1]} - {product[2]} грн",
            callback_data=f"buy_{product[0]}"
        ))
    bot.send_message(chat_id, "Оберіть товар:", reply_markup=markup)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Вітаємо у RoshenBot! Оберіть товар із меню.")
    show_menu(message.chat.id)

@bot.message_handler(commands=['menu'])
def menu_command(message):
    show_menu(message.chat.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('buy_'))
def handle_purchase(call):
    product_id = call.data.split('_')[1]
    conn = sqlite3.connect('db/roshen.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, price FROM products WHERE id = ?', (product_id,))
    product = cursor.fetchone()
    conn.close()
    if product:
        bot.send_message(call.message.chat.id, f"Ви обрали: {product[0]} за {product[1]} грн. Дякуємо за замовлення!")
    else:
        bot.send_message(call.message.chat.id, "Обраний товар не знайдено. Спробуйте ще раз.")

bot.polling(none_stop=True)
