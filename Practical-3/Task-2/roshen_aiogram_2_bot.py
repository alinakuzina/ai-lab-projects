from aiogram import Bot, Dispatcher, types
from aiogram.types import KeyboardButton
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram import F
import sqlite3
import logging
import stripe
from aiohttp import web
import asyncio
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext


# Initialize the bot
BOT_TOKEN = 'token in report'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Stripe API configuration
STRIPE_SECRET_KEY = 'token in report'
stripe.api_key = STRIPE_SECRET_KEY
WEBHOOK_SECRET = 'token in report' 

# Database initialization
def init_db():
    """Initialize the SQLite database and create the users table."""
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT,
                        password TEXT,
                        balance REAL DEFAULT 0.0
                    )''')
    conn.commit()
    conn.close()

init_db()

# Define states for registration
class Registration(StatesGroup):
    waiting_for_username = State()
    waiting_for_password = State()

# --- Telegram Bot Handlers ---

@dp.message(Command("start"))
async def start_command(message: types.Message):
    """Handle the /start command: register the user and show a welcome message."""
    await message.answer("Welcome! Use /help to see available commands.")

@dp.message(Command("help"))
async def help_command(message: types.Message):
    """Handle the /help command: list all available commands."""
    commands = "/start - Start the bot\n/register - Register your account\n/balance - Check your balance\n/topup - Top up your balance"
    await message.answer(f"Available commands:\n{commands}")

@dp.message(Command("register"))
async def start_registration(message: types.Message, state: FSMContext):
    """Start the registration process."""
    await message.answer("Please enter your username:")
    await state.set_state(Registration.waiting_for_username)

@dp.message(Registration.waiting_for_username)
async def process_username(message: types.Message, state: FSMContext):
    """Handle the username input."""
    await state.update_data(username=message.text)
    await message.answer("Please enter your password:")
    await state.set_state(Registration.waiting_for_password)

@dp.message(Registration.waiting_for_password)
async def process_password(message: types.Message, state: FSMContext):
    """Handle the password input and save the user."""
    data = await state.get_data()
    username = data.get("username")
    password = message.text
    user_id = message.from_user.id

    # Save to database
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (id, username, password) VALUES (?, ?, ?)", (user_id, username, password))
    conn.commit()
    conn.close()

    await message.answer("You have successfully registered!")
    await state.clear()

@dp.message(Command("balance"))
async def check_balance(message: types.Message):
    """Handle the /balance command: retrieve and display the user's balance."""
    user_id = message.from_user.id
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
    balance = cursor.fetchone()
    conn.close()
    if balance:
        await message.answer(f"Your current balance: {balance[0]:.2f} USD")
    else:
        await message.answer("You are not registered. Use /register to register.")

@dp.message(Command("topup"))
async def topup_balance(message: types.Message):
    """Handle the /topup command: provide options for top-up amounts."""
    builder = ReplyKeyboardBuilder()
    builder.add(KeyboardButton(text="10 USD"))
    builder.add(KeyboardButton(text="20 USD"))
    builder.add(KeyboardButton(text="50 USD"))
    builder.adjust(3)
    await message.answer("Choose the amount to top up:", reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message(F.text.in_(["10 USD", "20 USD", "50 USD"]))
async def process_topup(message: types.Message):
    """Handle top-up selection: create a Stripe payment session."""
    amount = int(message.text.split()[0])  # Extract amount
    user_id = message.from_user.id

    # Create a Stripe Checkout Session
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'usd',
                'product_data': {'name': f"Balance Top-Up {amount} USD"},
                'unit_amount': amount * 100,
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url=f'http://localhost:4242/success?user_id={user_id}&amount={amount}',
        cancel_url='http://localhost:4242/cancel',
        metadata={'user_id': user_id, 'amount': amount},
    )

    # Send the payment link to the user
    await message.answer(f"To top up, click the link below:\n{session.url}", disable_web_page_preview=True)

# --- Stripe Webhook Server ---

async def stripe_webhook(request):
    """Handle Stripe webhook events."""
    payload = await request.text()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, WEBHOOK_SECRET
        )
    except ValueError as e:
        # Invalid payload
        logging.error(f"Invalid payload: {e}")
        return web.Response(status=400)
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logging.error(f"Invalid signature: {e}")
        return web.Response(status=400)

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = int(session['metadata']['user_id'])
        amount = int(session['metadata']['amount'])

        # Update user balance
        conn = sqlite3.connect('bot.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (amount, user_id))
        conn.commit()
        conn.close()

        logging.info(f"User {user_id} balance updated by {amount} USD")

        # Send a success message to the user
        await bot.send_message(chat_id=user_id, text=f"Payment of {amount} USD received. Your balance has been updated.")

    return web.Response(status=200)

# --- Run Telegram Bot and Webhook Server ---

async def main():
    """Run the bot and the webhook server."""
    # Create aiohttp app for webhook server
    app = web.Application()
    app.router.add_post('/webhook', stripe_webhook)

    # Run both bot and webhook server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 4242)
    await site.start()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())