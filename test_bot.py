import pytest
from unittest.mock import MagicMock, AsyncMock
from aiogram import types, Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BufferedInputFile
from PIL import Image
from pathlib import Path
import torch
import asyncio
import os
import random
import sys

# Добавьте путь к вашему модулю
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import (  # Замените your_bot_module на имя вашего файла
    ImageStates,
    convert,
    load_model,
    cmd_start,
    instuction,
    get_example,
    info,
    start_image_upload,
    handle_style_choice,
    handle_image,
    handle_wrong_input,
    echo_message
)

# Фикстуры для тестирования
@pytest.fixture
def bot():
    mock_bot = MagicMock(spec=Bot)
    mock_bot.id = 123456
    return mock_bot

@pytest.fixture
def dispatcher():
    return Dispatcher(storage=MemoryStorage())

@pytest.fixture
def message():
    msg = MagicMock(spec=types.Message)
    msg.from_user = MagicMock()
    msg.from_user.id = 123
    msg.chat = MagicMock()
    msg.chat.id = 456
    msg.answer = AsyncMock()
    msg.answer_photo = AsyncMock()
    return msg

@pytest.fixture
def callback_query():
    cb = MagicMock(spec=types.CallbackQuery)
    cb.message = MagicMock()
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    cb.from_user = MagicMock()
    cb.from_user.id = 123
    return cb   

# Тесты для функций обработки изображений
def test_convert(mocker, tmp_path):
    # Создаем тестовое изображение
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (256, 256), color="red")
    img.save(img_path)
    
    output_path = tmp_path / "output.jpg"
    
    # Мокируем модель
    mock_net = mocker.MagicMock()
    mock_net.eval.return_value = None
    mock_net.return_value = torch.rand(1, 3, 256, 256) * 0.5 + 0.25
    
    # Вызываем функцию
    convert(str(img_path), str(output_path), mock_net)
    
    # Проверяем результаты
    assert os.path.exists(output_path)
    output_img = Image.open(output_path)
    assert output_img.size == (256, 256)

def test_load_model(mocker):
    mock_model = mocker.MagicMock()
    mock_load = mocker.patch("torch.load")
    
    load_model(mock_model, "test_model.pth")
    
    mock_model.load_state_dict.assert_called_once()
    mock_model.to.assert_called_once()
    mock_model.eval.assert_called_once()

# Тесты для обработчиков сообщений
@pytest.mark.asyncio
async def test_cmd_start(message):
    await cmd_start(message)
    message.answer.assert_called()
    assert "Выберите действие" in message.answer.call_args[0][0]
    
@pytest.mark.asyncio
async def test_handle_style_choice(callback_query, dispatcher):
    # Создаем мок бота
    mock_bot = MagicMock(spec=Bot)
    mock_bot.id = 123456
    
    state = dispatcher.fsm.get_context(bot=mock_bot, chat_id=456, user_id=123)
    callback_query.data = "sum2win"
    
    await handle_style_choice(callback_query, state)
    
    data = await state.get_data()
    assert data["net_type"] == "sum2win_G"
    assert await state.get_state() == ImageStates.waiting_for_image
    
    # Проверяем вызов с учетом reply_markup
    callback_query.message.answer.assert_called_once()
    args, kwargs = callback_query.message.answer.call_args
    assert args[0] == "Отлично, пришлите изображение"
    assert 'reply_markup' in kwargs
    
@pytest.mark.asyncio
async def test_handle_image_photo(message, bot, mocker, dispatcher, tmp_path):
    # Создаем временный входной файл
    input_path = tmp_path / "input.jpg"
    input_path.touch()
    
    # Создаем временный выходной файл
    output_path = tmp_path / "output.jpg"
    output_path.touch()
    
    # Мокируем convert, чтобы она просто создавала файл
    def mock_convert(in_path, out_path, net):
        with open(out_path, 'wb') as f:
            f.write(b'image_data')
    
    mocker.patch("bot.convert", side_effect=mock_convert)
    
    state = dispatcher.fsm.get_context(bot=bot, chat_id=456, user_id=123)
    await state.set_state(ImageStates.waiting_for_image)
    await state.update_data(net_type="sum2win_G")
    
    # Мокируем фото
    message.photo = [MagicMock(), MagicMock(file_id="test_photo")]
    message.document = None
    
    # Мокируем методы бота
    mock_file = MagicMock(file_path="test_path")
    bot.get_file = AsyncMock(return_value=mock_file)
    bot.download_file = AsyncMock(side_effect=lambda file_path, destination: Path(destination).touch())
    
    await handle_image(message, state, bot)
    
    # Проверяем вызовы
    bot.download_file.assert_called_once()
    message.answer.assert_any_call("Идет обработка изображения...")
    message.answer_photo.assert_called_once()

@pytest.mark.asyncio
async def test_handle_wrong_input(message, dispatcher):
    # Создаем мок бота
    mock_bot = MagicMock(spec=Bot)
    mock_bot.id = 123456
    
    state = dispatcher.fsm.get_context(bot=mock_bot, chat_id=456, user_id=123)
    await state.set_state(ImageStates.waiting_for_image)
    
    await handle_wrong_input(message)
    message.answer.assert_called_with('Пожалуйста, отправьте изображение или напишите "меню"')

@pytest.mark.asyncio
async def test_echo_message(message):
    message.text = "Test message"
    await echo_message(message)
    message.send_copy.assert_called_with(chat_id=message.chat.id)

# Тесты для callback обработчиков
@pytest.mark.asyncio
async def test_instuction(callback_query):
    await instuction(callback_query)
    
    # Проверяем первый вызов answer
    first_call_args = callback_query.message.answer.call_args_list[0][0][0]
    assert "Бот делает стилизацию" in first_call_args
    
    # Проверяем второй вызов answer
    second_call_args = callback_query.message.answer.call_args_list[1][0][0]
    assert "Вы также можете получить готовые примеры" in second_call_args
    
@pytest.mark.asyncio
async def test_get_example(callback_query, mocker):
    # Мокируем работу с файлами для двух папок
    mocker.patch("os.listdir", side_effect=[["file1.jpg", "file2.jpg"], ["file3.jpg"]])
    mocker.patch("os.path.isfile", return_value=True)
    
    # Мокируем BufferedInputFile
    mocker.patch("aiogram.types.BufferedInputFile", return_value="mocked_file")
    
    # Мокируем answer_photo как AsyncMock
    callback_query.message.answer_photo = AsyncMock()
    
    # Мокируем random.choice
    mocker.patch("random.choice", side_effect=lambda x: x[0])
    
    # Мокируем open для возврата тестовых данных
    mock_open = mocker.mock_open(read_data=b"test_image_data")
    mocker.patch("builtins.open", mock_open)
    
    await get_example(callback_query, MagicMock())
    
    # Проверяем вызовы
    assert callback_query.message.answer_photo.await_count == 2
@pytest.mark.asyncio
async def test_info(callback_query):
    await info(callback_query)
    callback_query.message.answer.assert_called_once()
    
    # Получаем аргументы вызова
    args, kwargs = callback_query.message.answer.call_args
    
    # Проверяем содержимое ответа
    assert "Stepik id" in kwargs.get('text', '')

@pytest.mark.asyncio
async def test_start_image_upload(callback_query):
    await start_image_upload(callback_query, MagicMock())
    callback_query.message.answer.assert_called()
    assert "выбрать один из режимов" in callback_query.message.answer.call_args[0][0]