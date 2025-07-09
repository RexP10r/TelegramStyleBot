# TelegramStyleBot Docs RU
Пользовательская документация телеграм бота со стилизацией на русском

## Разделы
- [Архитектура модели и обучение](ARCHITECTURE.md) 
- [Эксперименты](EXPERIMENTS.md)

## Результаты
|реальное изображение|сгенерированное|
|-|-|
|![](imgs/final1-1.jpg)|![](imgs/final1-2.jpg)|
|![](imgs/final2-1.jpg)|![](imgs/final2-2.jpg)|

## Быстрый старт
Склонируйте репозиторий на локальное хранилище
```bash
git clone https://github.com/Da47645/TelegramStyleBot.git
```
Соберите докер образ
```bash 
docker build -t my-bot .
```
Запуск
```bash
docker run -d -e TOKEN="your_bot_token" --name my-bot my-bot
```
Или создайте `.env` файл с токеном вашего бота
```
TOKEN=your_token
```
Тогда команда будет такой
```bash
docker run -d --name my-bot my-bot
```
Бот запустится в фоновом режиме, используйте следующие [команды](docker_guide.md) для взаимодействя с докером. 

