# CRPlayer Project Structure

## Architecture Overview
```
[Development Laptop] ←→ [GPU Server (GTX3060)]
       ↓                        ↓
   Development            All Critical Components:
   Monitoring             • Android Emulator (Headless)
   Configuration          • Computer Vision Pipeline
                         • ML Training/Inference
                         • Game Environment
                         • Data Storage
```

## Directory Structure
```
CRPlayer/
├── src/
│   ├── core/                   # Core RL components
│   │   ├── actor_critic.py     # Actor-Critic network
│   │   ├── environment.py      # Gym environment wrapper
│   │   ├── trainer.py          # Training loop
│   │   └── replay_buffer.py    # Experience replay
│   │
│   ├── vision/                 # Computer Vision
│   │   ├── screen_capture.py   # Screen capture from emulator
│   │   ├── state_parser.py     # Parse game state from frames
│   │   ├── object_detection.py # Detect units, towers, etc.
│   │   └── preprocessing.py    # Frame preprocessing
│   │
│   ├── emulator/               # Emulator control
│   │   ├── adb_controller.py   # ADB commands
│   │   ├── action_executor.py  # Execute game actions
│   │   └── emulator_setup.py   # Emulator configuration
│   │
│   ├── server/                 # Server components
│   │   ├── api_server.py       # REST API for control
│   │   ├── websocket_server.py # Real-time communication
│   │   └── monitoring.py       # Performance monitoring
│   │
│   └── client/                 # Client (laptop) components
│       ├── dashboard.py        # Training dashboard
│       ├── config_manager.py   # Configuration management
│       └── remote_control.py   # Remote server control
│
├── config/                     # Configuration files
│   ├── model_config.yaml       # Model hyperparameters
│   ├── training_config.yaml    # Training settings
│   └── server_config.yaml      # Server settings
│
├── data/                       # Data storage
│   ├── replays/               # Game replay data
│   ├── models/                # Saved model checkpoints
│   └── logs/                  # Training logs
│
├── scripts/                    # Utility scripts
│   ├── setup_emulator.sh      # Emulator setup script
│   ├── install_dependencies.sh # Dependencies installation
│   └── start_training.py      # Training starter script
│
├── tests/                      # Unit tests
│   ├── test_vision.py
│   ├── test_emulator.py
│   └── test_models.py
│
├── docker/                     # Docker configuration
│   ├── Dockerfile.server      # Server container
│   └── docker-compose.yml     # Multi-container setup
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── setup.py                  # Package setup
```

## Component Responsibilities

### Server (GTX3060)
- **Emulator Management**: Headless Android emulator
- **Computer Vision**: Real-time frame processing
- **ML Pipeline**: Training and inference
- **Game Control**: Action execution via ADB
- **Data Storage**: Replay buffer and model checkpoints

### Laptop (Development)
- **Code Development**: Main codebase editing
- **Experiment Management**: Configuration and monitoring
- **Remote Control**: Server management via SSH/API
- **Visualization**: Training metrics and dashboards

## Communication Protocol
- **SSH**: Secure server access and file transfer
- **REST API**: Configuration and control commands
- **WebSocket**: Real-time data streaming (frames, metrics)
- **VNC**: Optional visual access to emulator (debugging)
