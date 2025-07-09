## Little guide
```bash
# Логи в реальном времени
docker logs -f my-bot
# Логи с временными метками
docker logs -t my-bot
# Остановить
docker stop my-bot
# Если не останавливается
docker kill my-bot
# Запустить
docker start my-bot
# Перезапустить
docker restart my-bot
# Посмотреть процессы
docker ps -a
# Удалить
docker rm my-bot
# Вывод образов 
docker images
# Удалить все образы
docker rmi $(docker images -q)
```