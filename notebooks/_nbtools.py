"""
Narzedzie do budowania i WYKONYWANIA notebookow eksperymentow.

Uzycie (w skrypcie spec): make_and_run("NAZWA.ipynb", cells, timeout=...)
gdzie cells to lista krotek ("md"|"code", "zrodlo").

Notebook jest wykonywany w katalogu notebooks/ (cwd), wiec sciezki "../src" i
"../data" dzialaja jak w istniejacych notebookach. Zapisany .ipynb ma realne
outputy z faktycznego uruchomienia (gwarancja, ze kod dziala i dane sie zgadzaja).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient


NB_DIR = Path(__file__).resolve().parent


def make_and_run(name: str, cells: list[tuple[str, str]], timeout: int = 1800) -> Path:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    built = []
    for kind, source in cells:
        if kind == "md":
            built.append(new_markdown_cell(source))
        elif kind == "code":
            built.append(new_code_cell(source))
        else:
            raise ValueError(f"Nieznany typ komorki: {kind}")
    nb.cells = built

    out_path = NB_DIR / name
    print(f"[build] {name}: {len(built)} komorek", flush=True)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},  # cwd = notebooks/
        allow_errors=False,
    )
    t0 = time.time()
    print(f"[run] wykonuje {name} (timeout={timeout}s)...", flush=True)
    client.execute()
    nbformat.write(nb, out_path)
    print(f"[ok] zapisano {name} ({time.time()-t0:.0f}s)", flush=True)
    return out_path
