@echo off
REM Build script for CRPlayer Dashboard with Annotation Tool

echo Building CRPlayer Dashboard...

REM Ensure we're in the dashboard directory
cd /d "%~dp0"

REM Check if research models.py exists
if not exist "..\research\screen-page-classification\models.py" (
    echo Warning: Research models.py not found. Autolabel feature may not work.
    echo Expected at: ..\research\screen-page-classification\models.py
)

REM Check if model checkpoint exists
if not exist "..\data\models\SingleLabelClassification\model.pth" (
    echo Warning: Model checkpoint not found. Autolabel will not work until a model is trained.
    echo Expected at: ..\data\models\SingleLabelClassification\model.pth
)

REM Build and start the containers
echo Building Docker containers...
docker-compose build --no-cache

echo Starting services...
docker-compose up -d

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 5 /nobreak > nul

REM Check health
docker-compose ps

echo.
echo Dashboard is available at: http://localhost:8080
echo Annotation tool is available at: http://localhost:8080/annotation/
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
