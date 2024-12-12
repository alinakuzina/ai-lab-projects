from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
import sqlite3

# Bot initialization
BOT_TOKEN = 'token in report'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Database setup
def init_db():
    conn = sqlite3.connect('orders.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        product TEXT,
                        quantity INTEGER
                    )''')
    conn.commit()
    conn.close()

init_db()

# Start command
@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    main_menu = ReplyKeyboardMarkup(resize_keyboard=True)
    main_menu.add(KeyboardButton("🛍️ Place an Order"), KeyboardButton("ℹ️ Help"))
    await message.answer("Welcome to Roshen Bot! Choose an option below:", reply_markup=main_menu)

# Help command
@dp.message_handler(lambda message: message.text == "ℹ️ Help")
async def help_command(message: types.Message):
    commands = "🛍️ Place an Order - Start ordering products\nℹ️ Help - Show this help menu"
    await message.answer(f"Available commands:\n{commands}")

# Order command
@dp.message_handler(lambda message: message.text == "🛍️ Place an Order")
async def order_command(message: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(InlineKeyboardButton("🍫 Chocolate", callback_data="order_chocolate"),
                 InlineKeyboardButton("🍬 Candy", callback_data="order_candy"),
                 InlineKeyboardButton("🍪 Cookies", callback_data="order_cookies"))
    await message.answer("Please choose a product:", reply_markup=keyboard)

# Callback for ordering
@dp.callback_query_handler(lambda c: c.data.startswith('order_'))
async def process_order(callback_query: types.CallbackQuery):
    product = callback_query.data.split('_')[1]
    user_id = callback_query.from_user.id

    # Save order to the database
    conn = sqlite3.connect('orders.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (user_id, product, quantity) VALUES (?, ?, ?)", (user_id, product, 1))
    conn.commit()
    conn.close()

    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, f"✅ You have successfully ordered {product}!")

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)