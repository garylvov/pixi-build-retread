#!/usr/bin/env python3
"""Classify third-party imports in a source tree as SATISFIED or MISSING.

Deception vectors this explicitly defends against:
  * RELATIVE imports          -- `from . import x` / `from ..y import z`
                                 (ast.ImportFrom.level > 0) are intra-package,
                                 never a dependency. Dropped.
  * FIRST-PARTY absolute      -- a package importing its own top-level name
                                 (neural_wbc.core importing neural_wbc) looks
                                 third-party to a naive walker. Dropped by
                                 collecting the tree's own top-level package
                                 names first.
  * STDLIB                    -- sys.stdlib_module_names (3.10+), not a
                                 hand-maintained list.
  * MODULE != DISTRIBUTION    -- cv2/opencv-python, PIL/pillow,
                                 sklearn/scikit-learn. Resolved via
                                 importlib.metadata.packages_distributions()
                                 IN THE TARGET INTERPRETER, so the mapping is
                                 measured, not guessed.
  * CONDITIONAL imports       -- inside try/except or `if TYPE_CHECKING:` are
                                 still reported, but FLAGGED, because an
                                 optional import is not the same obligation.

Run with the interpreter of the environment being audited.
"""
import ast, sys, json
from collections import defaultdict
from pathlib import Path

def own_toplevel(root: Path) -> set[str]:
    """Top-level package/module names the tree itself defines."""
    names = set()
    for p in root.rglob("*"):
        if p.is_dir() and (p / "__init__.py").exists():
            names.add(p.name)
        # PEP 420 namespace dirs that contain packages
        if p.is_dir() and any((c / "__init__.py").exists() for c in p.iterdir() if c.is_dir()):
            names.add(p.name)
    for p in root.rglob("*.py"):
        if p.parent == root:
            names.add(p.stem)
    return names

def walk(root: Path, own: set[str]):
    hits = defaultdict(lambda: {"files": set(), "conditional": False})
    for f in root.rglob("*.py"):
        if any(part in {".git", "build", "dist", ".eggs", "__pycache__"} for part in f.parts):
            continue
        try:
            tree = ast.parse(f.read_text(errors="replace"), str(f))
        except SyntaxError:
            continue
        # Mark nodes that sit inside try/except or `if TYPE_CHECKING`.
        soft = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Try,)) or (
                isinstance(n, ast.If)
                and "TYPE_CHECKING" in ast.dump(n.test)
            ):
                for c in ast.walk(n):
                    soft.add(id(c))
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level and n.level > 0:      # RELATIVE -> never a dependency
                    continue
                if n.module:
                    mods = [n.module]
            for m in mods:
                top = m.split(".")[0]
                if not top or top in own:        # FIRST-PARTY
                    continue
                if top in sys.stdlib_module_names:
                    continue
                hits[top]["files"].add(str(f.relative_to(root)))
                if id(n) in soft:
                    hits[top]["conditional"] = True
    return hits

def main():
    root = Path(sys.argv[1]).resolve()
    own = own_toplevel(root)
    hits = walk(root, own)
    try:
        from importlib.metadata import packages_distributions
        prov = packages_distributions()
    except Exception:
        prov = {}
    import importlib.util as _u
    out = []
    for mod, info in sorted(hits.items()):
        dists = prov.get(mod)
        # SATISFACTION IS IMPORTABILITY, not metadata presence.
        # packages_distributions() returns None for conda-installed packages
        # (measured: numpy/scipy/tyro importable=1, packages_distributions=None
        # in the hover prefix), so keying on it reports false MISSING for every
        # conda-provided dependency. find_spec answers the real question;
        # packages_distributions is used only to NAME the distribution.
        try:
            importable = _u.find_spec(mod) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            importable = False
        out.append({
            "module": mod,
            "status": "SATISFIED" if importable else "MISSING",
            "importable": importable,
            "distribution": sorted(dists) if dists else None,
            "conditional": info["conditional"],
            "n_files": len(info["files"]),
            "example": sorted(info["files"])[0],
        })
    miss = [o for o in out if o["status"] == "MISSING"]
    print(f"### root={root}")
    print(f"### own top-level names ignored: {len(own)}")
    print(f"### third-party modules seen: {len(out)}   MISSING: {len(miss)}")
    for o in out:
        if o["status"] == "MISSING":
            flag = " [conditional]" if o["conditional"] else ""
            print(f"  MISSING    {o['module']:24s} files={o['n_files']:<4} e.g. {o['example']}{flag}")
    print(json.dumps(out, indent=None), file=sys.stderr)

main()
