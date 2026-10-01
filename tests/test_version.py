"""The version string is repeated in the build files; they must agree."""

import re
from pathlib import Path

import ndiffusion as nd

ROOT = Path(__file__).resolve().parent.parent


def _find(path, pattern):
    m = re.search(pattern, (ROOT / path).read_text(), re.M)
    assert m, f"no version found in {path}"
    return m.group(1)


def test_versions_agree():
    py = _find("pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    cmake = _find("CMakeLists.txt", r"^project\(\S+\s+VERSION\s+([0-9.]+)")
    doxygen = _find("Doxyfile", r"^PROJECT_NUMBER\s*=\s*(\S+)")
    assert nd.__version__ == py
    assert doxygen == py
    # CMake only accepts the numeric release, so 1.0.0.dev0 is 1.0.0 there.
    assert cmake == re.match(r"[0-9]+(\.[0-9]+)*", py).group(0)
