"""
Make the repository root importable to pytest.

CellNet ships no ``setup.py`` or ``pyproject.toml``, so ``cellnet`` is only
importable when the repository root is on ``sys.path``. ``python -m pytest``
puts it there; a bare ``pytest`` does not, and every test module then fails at
import with ``ModuleNotFoundError: No module named 'cellnet'``.

pytest imports the rootdir ``conftest.py`` before collecting anything, so doing
the insertion here explicitly makes both invocations work, independently of
pytest's import mode.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
