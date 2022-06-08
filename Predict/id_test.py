import telegram

token = '5560484908:AAE_ObxGlGoQD9UJ6gseYo14ImXXK-8X170'
bot = telegram.Bot(token=token)
updates = bot.getUpdates()
for u in updates:
    print(u.message)