# CRPlayer Deployment Guide

## Proxmox + LXC + Docker Setup

### Prerequisites
- Proxmox VE 7.x+ с NVIDIA GPU
- GTX3060 с установленными драйверами на хосте
- Доступ к root на Proxmox хосте

### Step 1: Подготовка Proxmox хоста

```bash
# Проверить GPU на хосте
nvidia-smi

# Убедиться что IOMMU включен
dmesg | grep -i iommu

# Установить необходимые пакеты
apt update
apt install -y pve-headers nvidia-driver-525
```

### Step 2: Создание LXC контейнера

```bash
# Скачать и запустить скрипт настройки
wget https://raw.githubusercontent.com/your-repo/CRPlayer/main/scripts/setup_proxmox_lxc.sh
chmod +x setup_proxmox_lxc.sh
./setup_proxmox_lxc.sh
```

### Step 3: Настройка в LXC контейнере

```bash
# Подключиться к контейнеру
pct enter 200

# Клонировать проект
git clone https://github.com/your-repo/CRPlayer.git
cd CRPlayer

# Проверить доступ к GPU
nvidia-smi

# Тест Docker с GPU
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### Step 4: Запуск сервисов

```bash
# Запустить все сервисы
docker-compose up -d

# Проверить статус
docker-compose ps

# Просмотр логов
docker-compose logs -f android-emulator
docker-compose logs -f ml-server
```

## Тестирование системы

### Test 1: Проверка эмулятора

```bash
# Подключение к эмулятору через ADB
adb connect localhost:5555
adb devices

# Проверка экрана через VNC
# Подключиться VNC клиентом к IP_КОНТЕЙНЕРА:5900
```

### Test 2: Проверка API сервера

```bash
# Тест REST API
curl http://localhost:8080/health

# Тест WebSocket (через браузер или wscat)
wscat -c ws://localhost:8080/ws
```

### Test 3: Проверка захвата экрана

```python
# Тестовый скрипт
from src.emulator.adb_controller import ADBController, EmulatorConfig
from src.vision.screen_capture import ScreenCapture, CaptureConfig

# Настройка
emulator_config = EmulatorConfig(host="localhost", port=5555)
capture_config = CaptureConfig(fps=10)

# Подключение
adb = ADBController(emulator_config)
adb.connect()

# Захват кадра
capture = ScreenCapture(adb, capture_config)
capture.start_capture()

frame = capture.get_latest_frame()
if frame is not None:
    print(f"Captured frame: {frame.shape}")
    capture.save_frame(frame, "test_frame.png")
else:
    print("Failed to capture frame")

capture.stop_capture()
```

## Мониторинг

### Grafana Dashboard
- URL: http://IP_КОНТЕЙНЕРА:3000
- Login: admin / crplayer123

### Полезные команды

```bash
# Мониторинг ресурсов
docker stats

# Логи эмулятора
docker-compose logs -f android-emulator

# Перезапуск сервиса
docker-compose restart ml-server

# Обновление кода
git pull
docker-compose build ml-server
docker-compose up -d ml-server
```

## Troubleshooting

### Проблема: GPU не доступен в контейнере
```bash
# Проверить на хосте
ls -la /dev/nvidia*

# Проверить в контейнере
lxc-device -n 200 list

# Перезапустить контейнер
pct stop 200 && pct start 200
```

### Проблема: Эмулятор не запускается
```bash
# Проверить KVM
ls -la /dev/kvm

# Проверить права доступа
docker-compose exec android-emulator ls -la /dev/kvm

# Логи эмулятора
docker-compose logs android-emulator
```

### Проблема: ADB не подключается
```bash
# Перезапустить ADB
adb kill-server
adb start-server

# Проверить порты
netstat -tlnp | grep 5555
```

## Оптимизация производительности

### Настройки LXC
```bash
# В /etc/pve/lxc/200.conf добавить:
lxc.cgroup2.memory.max = 12G
lxc.cgroup2.cpu.max = 600000 100000  # 6 cores
```

### Настройки эмулятора
```bash
# Увеличить RAM эмулятора
# В docker-compose.yml изменить:
environment:
  - EMULATOR_RAM=4096
```

### Настройки Docker
```bash
# Ограничить ресурсы контейнеров
# В docker-compose.yml добавить:
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```
