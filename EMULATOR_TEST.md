# Android Emulator Test Guide

## Быстрый запуск для тестирования

### Шаг 1: Сборка и запуск контейнера

```bash
# В директории CRPlayer
cd docker

# Запуск только эмулятора для тестирования
docker-compose -f docker-compose.emulator-test.yml up --build -d

# Просмотр логов запуска
docker-compose -f docker-compose.emulator-test.yml logs -f android-emulator-test
```

### Шаг 2: Ожидание загрузки эмулятора

Эмулятор может загружаться 2-5 минут. Следите за логами:

```bash
# Проверка статуса
docker-compose -f docker-compose.emulator-test.yml ps

# Проверка здоровья контейнера
docker inspect cr_emulator_test --format='{{.State.Health.Status}}'
```

### Шаг 3: Подключение через VNC

1. **Найдите IP адрес LXC контейнера:**
   ```bash
   hostname -I
   ```

2. **Подключитесь VNC клиентом к `IP_АДРЕС:5900`**
   - Windows: TightVNC, RealVNC, или встроенный Remote Desktop
   - macOS: встроенный Screen Sharing или VNC Viewer
   - Linux: Remmina, TigerVNC

3. **Пароль VNC:** `crplayer123` (если потребуется)

### Шаг 4: Автоматическое тестирование

```bash
# Запуск тестового скрипта
python3 scripts/test_emulator.py

# Или из контейнера
docker exec cr_emulator_test python3 /app/scripts/test_emulator.py
```

## Ожидаемые результаты

### ✅ Успешный запуск
- Контейнер в статусе `healthy`
- VNC показывает рабочий стол Android
- ADB подключается на порт 5555
- Тестовый скрипт проходит все проверки

### ❌ Возможные проблемы

#### Проблема: Контейнер не запускается
```bash
# Проверить логи
docker-compose -f docker-compose.emulator-test.yml logs android-emulator-test

# Проверить доступ к KVM
ls -la /dev/kvm

# Проверить права доступа
docker run --rm --privileged ubuntu:22.04 ls -la /dev/kvm
```

#### Проблема: VNC не подключается
```bash
# Проверить порты
netstat -tlnp | grep 5900

# Проверить процесс x11vnc в контейнере
docker exec cr_emulator_test ps aux | grep x11vnc
```

#### Проблема: Эмулятор не загружается
```bash
# Увеличить память эмулятора
# В docker-compose.emulator-test.yml изменить:
environment:
  - EMULATOR_RAM=4096

# Перезапустить
docker-compose -f docker-compose.emulator-test.yml down
docker-compose -f docker-compose.emulator-test.yml up -d
```

## Тестирование функциональности

### Тест 1: Базовое взаимодействие
1. Подключиться через VNC
2. Увидеть рабочий стол Android
3. Попробовать тапы и свайпы мышью

### Тест 2: ADB команды
```bash
# Подключение
adb connect localhost:5555

# Список устройств
adb devices

# Информация о системе
adb shell getprop ro.build.version.release

# Тестовый тап
adb shell input tap 500 500
```

### Тест 3: Захват экрана
```bash
# Сделать скриншот
adb shell screencap -p /sdcard/test.png

# Скачать на локальную машину
adb pull /sdcard/test.png ./test_screenshot.png

# Проверить размер файла
ls -la test_screenshot.png
```

## Подготовка к установке Clash Royale

### Установка Google Play Services
1. Через VNC открыть Settings
2. Найти Google → Add Account
3. Войти в Google аккаунт
4. Открыть Google Play Store

### Установка Clash Royale
1. В Play Store найти "Clash Royale"
2. Установить игру
3. Запустить и пройти обучение
4. Убедиться что игра работает стабильно

## Мониторинг производительности

```bash
# Использование ресурсов
docker stats cr_emulator_test

# Логи в реальном времени
docker logs -f cr_emulator_test

# Вход в контейнер для отладки
docker exec -it cr_emulator_test bash
```

## Очистка после тестирования

```bash
# Остановка и удаление контейнера
docker-compose -f docker-compose.emulator-test.yml down

# Удаление образа (если нужно)
docker rmi crplayer_android-emulator-test

# Очистка volumes
docker volume prune
```
