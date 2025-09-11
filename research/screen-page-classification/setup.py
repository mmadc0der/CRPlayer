"""
Setup script for screen page classification infrastructure.
Handles installation, configuration, and initial setup.
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def check_python_version():
  """Check if Python version is compatible."""
  if sys.version_info < (3, 8):
    console.print("[red]Python 3.8 or higher is required![/red]")
    sys.exit(1)

  console.print(f"[green]Python version: {sys.version}[/green]")


def install_requirements():
  """Install required packages."""
  console.print("[blue]Installing required packages...[/blue]")

  requirements_file = Path(__file__).parent / "requirements.txt"

  if not requirements_file.exists():
    console.print("[red]requirements.txt not found![/red]")
    return False

  try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    console.print("[green]Requirements installed successfully![/green]")
    return True
  except subprocess.CalledProcessError as e:
    console.print(f"[red]Failed to install requirements: {e}[/red]")
    return False


def create_directories():
  """Create necessary directories."""
  console.print("[blue]Creating directories...[/blue]")

  directories = ["data", "experiments", "models", "logs", "outputs"]

  base_dir = Path(__file__).parent

  for directory in directories:
    dir_path = base_dir / directory
    dir_path.mkdir(exist_ok=True)
    console.print(f"  Created: {dir_path}")


def check_annotation_api():
  """Check if annotation API is available."""
  console.print("[blue]Checking annotation API...[/blue]")

  try:
    import requests
    response = requests.get("http://localhost:5000/api/projects", timeout=5)
    if response.status_code == 200:
      console.print("[green]Annotation API is running![/green]")
      return True
  except Exception:
    pass

  console.print("[yellow]Annotation API is not running. Start it with:[/yellow]")
  console.print("  cd /workspace/tools/annotation && python app.py")
  return False


def run_tests():
  """Run basic tests to verify installation."""
  console.print("[blue]Running basic tests...[/blue]")

  try:
    # Test imports
    from models import ModelFactory
    from data_loader import DatasetConfig
    from trainer import ClassificationTrainer

    # Test model creation
    model = ModelFactory.create_model('resnet50', num_classes=10)
    console.print("[green]Model creation test passed![/green]")

    # Test configuration
    config = DatasetConfig()
    console.print("[green]Configuration test passed![/green]")

    return True

  except Exception as e:
    console.print(f"[red]Test failed: {e}[/red]")
    return False


def main():
  """Main setup function."""
  console.print(Panel("Screen Page Classification - Setup", style="bold blue"))

  with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console,
  ) as progress:

    # Check Python version
    task1 = progress.add_task("Checking Python version...", total=None)
    check_python_version()
    progress.update(task1, description="Python version check complete")

    # Install requirements
    task2 = progress.add_task("Installing requirements...", total=None)
    if not install_requirements():
      console.print("[red]Setup failed during requirements installation![/red]")
      return
    progress.update(task2, description="Requirements installed")

    # Create directories
    task3 = progress.add_task("Creating directories...", total=None)
    create_directories()
    progress.update(task3, description="Directories created")

    # Check annotation API
    task4 = progress.add_task("Checking annotation API...", total=None)
    check_annotation_api()
    progress.update(task4, description="API check complete")

    # Run tests
    task5 = progress.add_task("Running tests...", total=None)
    if not run_tests():
      console.print("[red]Setup failed during testing![/red]")
      return
    progress.update(task5, description="Tests passed")

  console.print(
    Panel(
      "Setup completed successfully!\n\n"
      "Next steps:\n"
      "1. Start the annotation API: cd /workspace/tools/annotation && python app.py\n"
      "2. Run data inspection: python main.py inspect\n"
      "3. Run experiments: python main.py experiment --dataset-id 1\n"
      "4. Check examples: python example_usage.py",
      style="bold green"))


if __name__ == "__main__":
  main()
