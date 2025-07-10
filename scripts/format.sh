#!/bin/bash

# Автоматический скрипт форматирования кода
# Запуск: ./scripts/format.sh

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

echo "Запуск isort..."
isort .

echo "Запуск black..."
black .

echo "Запуск flake8..."
flake8 . || {
    echo "Flake8 обнаружил проблемы"
    exit 0
}

echo "Все операции форматирования и проверки завершены"