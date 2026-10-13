#!/usr/bin/env python3
"""Print candidate MMseqs2 provenance as JSONL without changing files or running clustering.

A current script, package record or executable is not proof that a historical job used it.
Associate command/banner records with the deployed T15_xs output before claiming recovery.
Uses only the standard library, so it can run through `ssh HOST 'python3 - ...' < this_file`.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

PATTERN = re.compile(r"mmseqs|easy[-_]cluster|linclust|cdhit_splits|T15_xs|min-seq-id|cov-mode|cluster-coverage", re.I)
TEXT_SUFFIXES = {".py", ".sh", ".sbatch", ".log", ".out", ".err", ".json", ".yaml", ".yml", ".cmd", ".params"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "site-packages", "pkgs"}


def emit(kind, **fields):
    print(json.dumps({"kind": kind, **fields}, ensure_ascii=False), flush=True)


def metadata(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def scan_text(path, context=3):
    try:
        info = metadata(path)
        if not info["bytes"]:
            if "_cluster_mmseqs" in path.parts:
                emit("empty_cache_file", **info)
            return
        previous = deque(maxlen=context)
        remaining = 0
        last_emitted = 0
        with path.open(encoding="utf-8", errors="replace") as source:
            for number, line in enumerate(source, 1):
                matched = bool(PATTERN.search(line))
                if matched:
                    for old_number, old_line in previous:
                        if old_number > last_emitted:
                            emit("text_context", **info, line=old_number, text=old_line)
                            last_emitted = old_number
                    remaining = context + 1
                if remaining and number > last_emitted:
                    emit("text_match" if matched else "text_context", **info,
                         line=number, text=line.rstrip("\r\n"))
                    last_emitted = number
                    remaining -= 1
                previous.append((number, line.rstrip("\r\n")))
    except OSError as exc:
        emit("read_error", path=str(path), error=str(exc))


def package_record(path):
    try:
        data = json.loads(path.read_text())
        fields = {key: data[key] for key in ("name", "version", "build", "build_number", "channel", "subdir", "sha256", "md5") if key in data}
        emit("installed_package_record_not_job_association", **metadata(path), package=fields)
    except (OSError, ValueError) as exc:
        emit("read_error", path=str(path), error=str(exc))


def executable_record(path):
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        emit("current_executable_not_historical_version", **metadata(path),
             resolved_path=str(path.resolve()), sha256=digest.hexdigest(),
             version_command=[str(path), "version"])
    except OSError as exc:
        emit("read_error", path=str(path), error=str(exc))


def scan_environment(prefix):
    conda_meta = prefix / "conda-meta"
    if not conda_meta.is_dir():
        return
    try:
        paths = sorted(conda_meta.iterdir())
    except OSError as exc:
        emit("read_error", path=str(conda_meta), error=str(exc))
        return
    for path in paths:
        if "mmseq" in path.name.lower() and path.suffix == ".json":
            package_record(path)
        elif path.name == "history":
            scan_text(path)
    binary = prefix / "bin/mmseqs"
    if binary.is_file():
        executable_record(binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original project or recovered project directory")
    parser.add_argument("--environment-root", type=Path, action="append", default=[], help="an environment prefix or directory containing prefixes; repeat as needed")
    parser.add_argument("--history-file", type=Path, action="append", default=[], help="shell history to search for matching commands only; repeat as needed")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("project root does not exist: " + str(root))
    emit("scope", root=str(root), read_only=True,
         interpretation="Candidate evidence only. File timestamps and current installed versions do not establish which command/version created the deployed split.")
    for directory, children, names in os.walk(root, onerror=lambda exc: emit("walk_error", error=str(exc))):
        children[:] = sorted(name for name in children if name not in SKIP_DIRS)
        parent = Path(directory)
        if parent.name == "conda-meta":
            scan_environment(parent.parent)
            children[:] = []
            continue
        for name in sorted(names):
            path = parent / name
            try:
                if path.is_symlink():
                    emit("symbolic_link", path=str(path), target=os.readlink(path), target_exists=path.exists())
                    if not path.exists():
                        continue
                if name == "mmseqs" and path.is_file():
                    executable_record(path)
                elif (path.suffix.lower() in TEXT_SUFFIXES or name in {"Snakefile", "history"}
                      or (".snakemake" in path.parts and parent.name == "metadata")):
                    scan_text(path)
                elif "_cluster_mmseqs" in path.parts:
                    emit("cache_member", **metadata(path))
                    if path.suffix == ".source":
                        scan_text(path)
            except OSError as exc:
                emit("read_error", path=str(path), error=str(exc))
    for environment_root in args.environment_root:
        environment_root = environment_root.expanduser()
        if not environment_root.is_dir():
            emit("environment_root_absent", path=str(environment_root))
            continue
        scan_environment(environment_root)
        try:
            children = sorted(environment_root.iterdir())
        except OSError as exc:
            emit("read_error", path=str(environment_root), error=str(exc))
            continue
        for child in children:
            if child.is_dir():
                scan_environment(child)
    for history in args.history_file:
        history = history.expanduser()
        if history.is_file():
            scan_text(history, context=0)
        else:
            emit("history_file_absent", path=str(history))
    emit("complete", historical_command_or_version_inferred=False)


if __name__ == "__main__":
    main()
