"""
Architectural tests: validate the import constraints from .importlinter
by walking the import graph manually.

Redundant with import-linter for safety. Both should agree.

Run with: pytest tests/test_architecture.py
"""
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / "src" / "bsig"


def imports_from(file_path: Path) -> set[str]:
    """Extract all imports from a Python file as fully-qualified names."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return set()
    tree = ast.parse(file_path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def all_imports_in_module(module_dir: Path) -> set[str]:
    """Aggregate imports across every .py file in a directory tree."""
    imports: set[str] = set()
    for py in module_dir.rglob("*.py"):
        imports |= imports_from(py)
    return imports


def test_core_does_not_import_from_other_layers() -> None:
    imports = all_imports_in_module(SRC / "core")
    forbidden = {"bsig.adapters", "bsig.medqa", "bsig.reference"}
    violations = {imp for imp in imports if any(imp.startswith(f) for f in forbidden)}
    assert not violations, f"core imports from forbidden layers: {violations}"


def test_adapters_do_not_import_domain_packs() -> None:
    imports = all_imports_in_module(SRC / "adapters")
    forbidden = {"bsig.medqa", "bsig.reference"}
    violations = {imp for imp in imports if any(imp.startswith(f) for f in forbidden)}
    assert not violations, f"adapters imports from forbidden modules: {violations}"


def test_reference_does_not_import_domain_packs() -> None:
    imports = all_imports_in_module(SRC / "reference")
    forbidden = {"bsig.medqa"}
    violations = {imp for imp in imports if any(imp.startswith(f) for f in forbidden)}
    assert not violations, f"reference imports from domain packs: {violations}"
