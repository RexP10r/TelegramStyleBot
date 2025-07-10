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
git clone https://github.com/RexP10r/TelegramStyleBot.git
cd TelegramStyleBot
```
Соберите докер  (6.29 гб)
```bash 
docker build -t my-bot .
```
Запуск передачей токена через переменную окружения:
```bash
docker run -d -e TOKEN="your_bot_token" --name my-bot my-bot
```
Запуск через .env файл (создайте в корне проекта):
```
TOKEN=your_token    # содержимое .env файла
```
```bash
docker run -d --name my-bot my-bot
```
Бот запустится в фоновом режиме, используйте следующие [команды](docker_guide.md) для взаимодействя с докером. 

