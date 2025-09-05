"""
Factory classes for creating test data using factory-boy.
"""

import factory
from typing import Dict, Any
from datetime import datetime
import json


class SessionMetadataFactory(factory.DictFactory):
    """Factory for creating session metadata."""
    game_name = factory.Sequence(lambda n: f"TestGame{n}")
    start_time = factory.LazyFunction(lambda: datetime.now().isoformat())
    version = "1.0.0"
    resolution = "1920x1080"
    fps = 30
    device_info = factory.Dict({
        "model": "TestDevice",
        "os": "TestOS 1.0"
    })


class ProjectDataFactory(factory.DictFactory):
    """Factory for creating project data."""
    name = factory.Sequence(lambda n: f"Test Project {n}")
    description = factory.LazyAttribute(lambda obj: f"Description for {obj.name}")


class DatasetDataFactory(factory.DictFactory):
    """Factory for creating dataset data."""
    name = factory.Sequence(lambda n: f"Test Dataset {n}")
    description = factory.LazyAttribute(lambda obj: f"Description for {obj.name}")
    target_type_id = 1  # SingleLabelClassification


class FrameDataFactory(factory.DictFactory):
    """Factory for creating frame data."""
    frame_id = factory.Sequence(lambda n: f"frame_{n:03d}")
    ts_ms = factory.Sequence(lambda n: n * 1000)  # 1 second intervals


class AnnotationDataFactory(factory.DictFactory):
    """Factory for creating annotation data."""
    game_state = factory.Iterator(["menu", "loading", "battle", "final"])
    importance = factory.Iterator([1, 2, 3])
    confidence = factory.Faker('pyfloat', left_digits=0, right_digits=2, positive=True, max_value=1.0)
    notes = factory.Faker('sentence')


class SaveRegressionRequestFactory(factory.DictFactory):
    """Factory for creating SaveRegressionRequest data."""
    session_id = factory.Sequence(lambda n: f"test_session_{n:03d}")
    dataset_id = 1
    frame_idx = factory.Sequence(lambda n: n)
    value = factory.Faker('pyfloat', left_digits=3, right_digits=3, positive=True, max_value=100.0)


class SaveSingleLabelRequestFactory(factory.DictFactory):
    """Factory for creating SaveSingleLabelRequest data."""
    session_id = factory.Sequence(lambda n: f"test_session_{n:03d}")
    dataset_id = 1
    frame_idx = factory.Sequence(lambda n: n)
    class_id = factory.Iterator([1, 2, 3, 4])


class SaveMultilabelRequestFactory(factory.DictFactory):
    """Factory for creating SaveMultilabelRequest data."""
    session_id = factory.Sequence(lambda n: f"test_session_{n:03d}")
    dataset_id = 1
    frame_idx = factory.Sequence(lambda n: n)
    class_ids = factory.List([factory.Iterator([1, 2, 3, 4]) for _ in range(2)])


class DatasetSessionSettingsFactory(factory.DictFactory):
    """Factory for creating dataset-session settings."""
    auto_advance = True
    show_confidence = False
    default_importance = 2
    hotkeys = factory.Dict({
        "1": "menu",
        "2": "loading", 
        "3": "battle",
        "4": "final"
    })
    custom_field = factory.Faker('word')


# Utility functions for creating complex test scenarios
def create_session_with_frames(session_id: str = None, frame_count: int = 5) -> Dict[str, Any]:
    """Create a session with associated frames."""
    if session_id is None:
        session_id = f"test_session_{factory.Faker('random_int', min=100, max=999).generate()}"
    
    metadata = SessionMetadataFactory()
    frames = [FrameDataFactory(frame_id=f"frame_{i:03d}") for i in range(1, frame_count + 1)]
    
    return {
        "session_id": session_id,
        "metadata": metadata,
        "frames": frames,
        "root_path": f"/test/data/{session_id}"
    }


def create_project_with_datasets(project_name: str = None, dataset_count: int = 2) -> Dict[str, Any]:
    """Create a project with multiple datasets."""
    if project_name is None:
        project_name = f"Test Project {factory.Faker('random_int', min=100, max=999).generate()}"
    
    project = ProjectDataFactory(name=project_name)
    datasets = []
    
    for i in range(dataset_count):
        dataset = DatasetDataFactory(
            name=f"{project_name} Dataset {i+1}",
            target_type_id=factory.Iterator([0, 1, 2]).generate()  # Random target type
        )
        datasets.append(dataset)
    
    return {
        "project": project,
        "datasets": datasets
    }


def create_annotation_scenario(session_count: int = 2, frames_per_session: int = 10) -> Dict[str, Any]:
    """Create a complete annotation scenario with sessions, projects, and datasets."""
    # Create project and dataset
    project_data = create_project_with_datasets(dataset_count=1)
    
    # Create sessions with frames
    sessions = []
    for i in range(session_count):
        session = create_session_with_frames(
            session_id=f"scenario_session_{i+1:03d}",
            frame_count=frames_per_session
        )
        sessions.append(session)
    
    return {
        "project": project_data["project"],
        "dataset": project_data["datasets"][0],
        "sessions": sessions
    }