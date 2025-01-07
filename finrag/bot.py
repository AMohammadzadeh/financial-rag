from telethon import TelegramClient, events
from finrag.config import settings
from finrag.chat import build_graph
from loguru import logger

api_id = settings.TELEGRAM_API_ID
api_hash = settings.TELEGRAM_API_HASH
bot_token = settings.TELEGRAM_BOT_TOKEN

client = TelegramClient('bot_session', api_id, api_hash).start(bot_token=bot_token)

graph = build_graph()

@client.on(events.NewMessage)
async def handle_message(event):
    user_message = event.message.message.strip()

    if user_message.lower() in ["/start", "hi", "hello"]:
        await event.reply("سلام! هر سؤال مالی‌ای که داری می‌تونی از من بپرسی.")
        return

    try:
        response = graph.invoke({"question": user_message})
        answer = response.get("answer", "متاسفم. برای این سؤال جوابی پیدا نکردم.")
    except Exception as e:
        answer = f"An error occurred while processing your query: {str(e)}"

    await event.reply(answer)

if __name__ == "__main__":
    logger.info("Bot is running...")
    client.run_until_disconnected()
