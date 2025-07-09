import asyncio
import logging
import os
import random
import sys
import tempfile

import torch
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.utils.formatting import (
    Bold,
    as_list,
    as_marked_section,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.token import TokenValidationError
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

from CycleGan import Generator

torch.set_default_dtype(torch.float32)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)
print("Default tensor type:", torch.get_default_dtype())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
noise_sources = [
    "aiogram.event",
    "aiogram.dispatcher.dispatcher",
    "aiogram.updates",
]

for source in noise_sources:
    logging.getLogger(source).setLevel(logging.WARNING)

logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

logging.getLogger(__name__).setLevel(logging.DEBUG)

load_dotenv(".env")

TOKEN = os.getenv("TOKEN")

if not TOKEN:
    raise ValueError("TOKEN не установлен")

bot = Bot(token=TOKEN)

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


class ImageStates(StatesGroup):
    waiting_for_image = State()
    format_selection = State()


path = "st_dicts/"

sum2win_netG_A2B = Generator(device, 3, 3)
sum2win_netG_B2A = Generator(device, 3, 3)

global net_type


def load_model(model, filename):
    model_path = path + filename
    model.load_state_dict(
        torch.load(model_path, map_location=device), strict=False
    )
    model.to(device)
    model.eval()
    print(f"Модель {filename} успешно загружена")


load_model(sum2win_netG_A2B, "sum2win_netG_A2B.pth")
load_model(sum2win_netG_B2A, "sum2win_netG_B2A.pth")


def convert(input_path, output_path, net):
    image = Image.open(input_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        net.eval()
        output_tensor = net(image_tensor)

        output_tensor = output_tensor.squeeze(0).to(device)
        output_tensor = output_tensor * 0.5 + 0.5  # [0.5, 0.5, 0.5] -> [0, 1]
        output_image = transforms.ToPILImage()(output_tensor)

        output_image.save(output_path)


@dp.message(F.text.func(lambda text: text and text.lower() == "меню"))
@dp.message(Command("start"))
async def cmd_start(message: Message):
    builder = InlineKeyboardBuilder()
    buttons = [
        ("Инструкция", "info"),
        ("Стилизация", "style"),
        ("Автор", "autor_info"),
    ]

    for text, callback_data in buttons:
        builder.button(text=text, callback_data=callback_data)

    builder.adjust(1)

    await message.answer(
        "Опять работать...", reply_markup=ReplyKeyboardRemove()
    )
    await message.answer(
        "Выберите действие:", reply_markup=builder.as_markup()
    )


@dp.callback_query(F.data == "info")
async def instuction(callback: types.CallbackQuery):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Меню")]], resize_keyboard=True
    )
    await callback.message.answer(
        "Бот делает стилизацию летних пейзажей в зимние и наоборот. "
        "Вам необходимо отправить фотографию размером 256x256 или\n"
        "с соотношением сторон 1:1, но тогда её размеронсть уменьшится "
        "до упомянутой выше.",
        reply_markup=keyboard,
    )

    builder = InlineKeyboardBuilder()
    builder.button(text="Прислать примеры", callback_data="example")
    builder.adjust(1)

    await callback.message.answer(
        "Вы также можете получить готовые примеры, чтобы прислать их боту.",
        reply_markup=builder.as_markup(),
    )


@dp.callback_query(F.data == "example")
async def get_example(callback: types.CallbackQuery, state: FSMContext):
    base_dir = "examples/"
    ex_dirs = ["summer", "winter"]
    captions = ["Летний пейзаж", "Зимний пейзаж"]

    for i in range(2):
        folder_path = os.path.join(base_dir, ex_dirs[i])

        files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        random_file = random.choice(files)
        file_path = os.path.join(folder_path, random_file)

        with open(file_path, "rb") as f:
            await callback.message.answer_photo(
                BufferedInputFile(f.read(), filename=random_file),
                caption=captions[i],
            )
    logger.info(f"Examples sent to chat {callback.message.chat.id}")


@dp.callback_query(F.data == "autor_info")
async def info(callback: types.CallbackQuery):
    content = as_list(
        as_marked_section(
            Bold("Stepik id:"),
            "-601740121",
        ),
        as_marked_section(
            Bold("Telegram:"),
            "-@DmGexNetwork",
        ),
        sep="\n",
    )
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Меню")]], resize_keyboard=True
    )
    await callback.message.answer(**content.as_kwargs(), reply_markup=keyboard)


@dp.callback_query(F.data == "style")
async def start_image_upload(callback: types.CallbackQuery, state: FSMContext):
    builder = InlineKeyboardBuilder()
    buttons = [
        ("лето->зима", "sum2win"),
        ("зима->лето", "win2sum"),
    ]

    for text, callback_data in buttons:
        builder.button(text=text, callback_data=callback_data)

    builder.adjust(1)

    await callback.message.answer(
        "Сейчас вам надо выбрать один из режимов:",
        reply_markup=builder.as_markup(),
    )


@dp.callback_query(F.data.in_(["sum2win", "win2sum"]))
async def handle_style_choice(
    callback: types.CallbackQuery, state: FSMContext
):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Меню")]], resize_keyboard=True
    )
    names_list = ["sum2win", "win2sum"]

    if callback.data in names_list:
        net_type = f"{callback.data}_G"

    await state.update_data(net_type=net_type)
    await state.set_state(ImageStates.waiting_for_image)
    await callback.message.answer(
        "Отлично, пришлите изображение", reply_markup=keyboard
    )
    await callback.answer()


@dp.message(ImageStates.waiting_for_image, F.photo | F.document)
async def handle_image(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    net_type = data.get("net_type", "sum2win_G")
    user_id = message.from_user.id

    if message.photo:
        file_id = message.photo[-1].file_id
        logger.info(f"Photo received from user {user_id}")
    elif message.document:
        mime_type = message.document.mime_type
        if not mime_type or not mime_type.startswith("image/"):
            await message.answer(
                "Пожалуйста, отправьте изображение в формате JPEG, PNG и т.п."
            )
            return
        file_id = message.document.file_id
        logger.info(f"Document image received from user {user_id}")
    else:
        await message.answer("Неподдерживаемый формат файла")
        return

    file = await bot.get_file(file_id)
    input_path = tempfile.mktemp(suffix=".jpg")
    output_path = tempfile.mktemp(suffix=".jpg")

    try:
        await bot.download_file(file.file_path, destination=input_path)
        await message.answer("Идет обработка изображения...")

        cur_net = None
        if net_type == "sum2win_G":
            cur_net = sum2win_netG_A2B
        elif net_type == "win2sum_G":
            cur_net = sum2win_netG_B2A

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, convert, input_path, output_path, cur_net
        )

        with open(output_path, "rb") as f:
            await message.answer_photo(
                BufferedInputFile(f.read(), filename="result.jpg"),
                caption="Результат стилизации",
            )

        logger.info(f"Image converted in chat {message.chat.id}")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await message.answer(f"Ошибка при обработке: {str(e)}")

    finally:
        for path in [input_path, output_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Can't delete temp file: {e}")

    await state.clear()


@dp.message(ImageStates.waiting_for_image)
async def handle_wrong_input(message: Message):
    await message.answer(
        'Пожалуйста, отправьте изображение или напишите "меню" '
    )


@dp.message()
async def echo_message(message: types.Message):
    try:
        await message.send_copy(chat_id=message.chat.id)
        logger.info(
            f"Echoed message from {message.from_user.id}"
            f"in chat {message.chat.id}"
        )
    except TelegramAPIError as e:
        logger.error(f"Telegram API Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def main():
    try:
        if not await bot.get_me():
            raise TokenValidationError("Invalid bot token")
        logger.info("Starting bot...")
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except TokenValidationError as e:
        logger.error(f"Token validation error: {e}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    finally:
        loop.close()
