"""Shim module to expose the annotation Flask app at top-level import 'app'.
Creates lightweight stubs for heavy optional dependencies (torch, psutil)
so that unit tests can run in minimal environments.
"""

from types import ModuleType
import sys
import importlib


# ---------------------------------------------------------------------------
# Provide stub modules for optional heavy dependencies to keep import light.
# ---------------------------------------------------------------------------

def _install_torch_stub():
    torch_stub = ModuleType("torch")

    # Minimal API used in the codebase
    def _not_available(*_a, **_kw):
        raise RuntimeError("Stubbed torch module: functionality not available in test env")

    torch_stub.cuda = ModuleType("torch.cuda")
    torch_stub.cuda.is_available = lambda: False
    torch_stub.cuda.memory_allocated = lambda: 0
    torch_stub.cuda.get_device_name = lambda *_a, **_kw: "cpu"

    # Provide a dummy Tensor type for isinstance checks
    class _Tensor:  # noqa: D401, N801
        pass

    torch_stub.Tensor = _Tensor
    torch_stub.device = lambda *_a, **_kw: "cpu"
    torch_stub.nn = ModuleType("torch.nn")
    torch_stub.nn.functional = ModuleType("torch.nn.functional")
    torch_stub.nn.functional.interpolate = _not_available

    sys.modules["torch"] = torch_stub


def _install_psutil_stub():
    psutil_stub = ModuleType("psutil")
    psutil_stub.cpu_percent = lambda interval=None: 0.0

    class _Mem:  # noqa: D401, N801
        percent = 0.0

    psutil_stub.virtual_memory = lambda: _Mem()
    sys.modules["psutil"] = psutil_stub


# Generic simple stub factory
def _install_simple_stub(name: str, attrs: list[str] | None = None):
    stub = ModuleType(name)
    if attrs:
        for attr in attrs:
            setattr(stub, attr, lambda *args, **kwargs: None)
    sys.modules[name] = stub


# Stub additional heavy optional deps if missing
for _mod, _attrs in {
    "numpy": ["array", "ndarray"],
    "cv2": ["imread", "imwrite", "resize"],
    "av": [],
}.items():
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ModuleNotFoundError:
            _install_simple_stub(_mod, _attrs)


# ---------------------------------------------------------------------------
# Alias nested annotation core package to top-level 'core' expected by legacy
# code so that `import core.session_manager` works without modifying sources.
# ---------------------------------------------------------------------------

if 'core' not in sys.modules:
    try:
        annotation_core = importlib.import_module('tools.annotation.core')
        sys.modules['core'] = annotation_core
        # Expose submodules manually if available
        try:
            session_manager_mod = importlib.import_module('tools.annotation.core.session_manager')
            sys.modules['core.session_manager'] = session_manager_mod
        except ModuleNotFoundError:
            pass
    except ModuleNotFoundError:
        # Fallback simple stub if annotation core missing
        _install_simple_stub('core')


if "torch" not in sys.modules:
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        _install_torch_stub()

if "psutil" not in sys.modules:
    try:
        import psutil  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        _install_psutil_stub()


# Finally import the real Flask app from the annotation package
from tools.annotation.app import app  # type: ignore  # noqa: E402,F401