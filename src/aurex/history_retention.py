"""Bounded, crash-safe retention for operator-created history snapshots.

Only immediate children of the configured history directory are managed.  The
live SQLite database, its WAL, the cache tree and registered task artifacts are
deliberately outside this component's authority.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat as stat_module
import tarfile
import threading
import time
import uuid


GIB = 1024 ** 3
_ARCHIVE_SUFFIX = ".tar.gz"
_LOCK_NAME = ".retention.lock"
_TEMP_PREFIX = ".retention-"
_MARKER_NAME = ".aurex-history-v1"
_MARKER_CONTENT = b"Aurex managed history directory v1\n"


def _allocated_size(path: Path) -> int:
    """Return real allocated bytes where available, without following links."""
    stat = path.lstat()
    blocks = getattr(stat, "st_blocks", None)
    return int(blocks * 512 if blocks is not None else stat.st_size)


def _tree_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file() or os.path.ismount(path):
        return _allocated_size(path)
    total = _allocated_size(path)
    device = path.lstat().st_dev
    for root, directories, files in os.walk(path, followlinks=False):
        base = Path(root)
        for name in list(directories):
            child = base / name
            stat = child.lstat()
            if child.is_symlink() or stat.st_dev != device or os.path.ismount(child):
                directories.remove(name)
                continue
            total += _allocated_size(child)
        for name in files:
            child = base / name
            if child.is_symlink() or child.lstat().st_dev != device:
                continue
            total += _allocated_size(child)
    return total


def _tree_mtime(path: Path) -> float:
    """Newest metadata timestamp in one snapshot, without following links."""
    newest = path.lstat().st_mtime
    if path.is_symlink() or path.is_file() or os.path.ismount(path):
        return newest
    device = path.lstat().st_dev
    for root, directories, files in os.walk(path, followlinks=False):
        base = Path(root)
        for name in list(directories):
            child = base / name
            stat = child.lstat()
            if child.is_symlink() or stat.st_dev != device or os.path.ismount(child):
                directories.remove(name)
                continue
            newest = max(newest, stat.st_mtime)
        for name in files:
            child = base / name
            if child.is_symlink() or child.lstat().st_dev != device:
                continue
            newest = max(newest, child.lstat().st_mtime)
    return newest


def _snapshot(path: Path) -> tuple[tuple[str, str, int, int], ...]:
    """Capture stable metadata while refusing to traverse symlinks."""
    root = path.parent
    device = path.lstat().st_dev
    rows: list[tuple[str, str, int, int]] = []

    def visit(item: Path) -> None:
        stat = item.lstat()
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            rows.append((relative, "link", 0, stat.st_mtime_ns))
            return
        if item.is_dir():
            if item != path and (stat.st_dev != device or os.path.ismount(item)):
                raise OSError(f"history snapshot crosses a filesystem boundary: {relative}")
            rows.append((relative, "dir", 0, stat.st_mtime_ns))
            with os.scandir(item) as listing:
                children = sorted((Path(entry.path) for entry in listing), key=lambda child: child.name)
            for child in children:
                visit(child)
            return
        if item.is_file():
            if stat.st_dev != device:
                raise OSError(f"history snapshot crosses a filesystem boundary: {relative}")
            rows.append((relative, "file", stat.st_size, stat.st_mtime_ns))
            return
        raise OSError(f"unsupported history entry: {relative}")

    visit(path)
    return tuple(rows)


def _source_signature(path: Path) -> tuple[tuple[str, str, int, str], ...]:
    """Hash a source snapshot for recovery from archive+source crash state."""
    root = path.parent
    device = path.lstat().st_dev
    rows: list[tuple[str, str, int, str]] = []

    def visit(item: Path) -> None:
        stat = item.lstat()
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            rows.append((relative, "link", 0, os.readlink(item)))
            return
        if item.is_dir():
            if item != path and (stat.st_dev != device or os.path.ismount(item)):
                raise OSError(f"history snapshot crosses a filesystem boundary: {relative}")
            rows.append((relative, "dir", 0, ""))
            with os.scandir(item) as listing:
                children = sorted((Path(entry.path) for entry in listing), key=lambda child: child.name)
            for child in children:
                visit(child)
            return
        if item.is_file():
            if stat.st_dev != device:
                raise OSError(f"history snapshot crosses a filesystem boundary: {relative}")
            digest, size = hashlib.sha256(), 0
            with item.open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            rows.append((relative, "file", size, digest.hexdigest()))
            return
        raise OSError(f"unsupported history entry: {relative}")

    visit(path)
    return tuple(sorted(rows))


def _normalized_link(parts: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                raise OSError("history archive link escapes its snapshot root")
            normalized.pop()
        else:
            normalized.append(part)
    return tuple(normalized)


def _verify_link(member_path: PurePosixPath, linkname: str, expected_root: str,
                 *, hardlink: bool) -> None:
    if not linkname:
        raise OSError("history archive contains an empty link target")
    target = PurePosixPath(linkname)
    if target.is_absolute():
        raise OSError("history archive contains an absolute link target")
    parts = target.parts if hardlink else (*member_path.parent.parts, *target.parts)
    normalized = _normalized_link(parts)
    if not normalized or normalized[0] != expected_root:
        raise OSError("history archive link escapes its snapshot root")


def _open_regular_nofollow(path: Path):
    """Open one private regular file and bind the descriptor to its pathname."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if (not stat_module.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_nlink != 1
                or identity != (path_stat.st_dev, path_stat.st_ino)):
            raise OSError("history archive must be one real private regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        return stream, identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _verify_archive_with_identity(
        path: Path, expected_root: str,
) -> tuple[tuple[tuple[str, str, int, str], ...], tuple[int, int]]:
    """Read every payload through an O_NOFOLLOW descriptor and return its inode."""
    found_root = False
    rows: list[tuple[str, str, int, str]] = []
    seen: set[str] = set()
    input_file, identity = _open_regular_nofollow(path)
    with input_file:
        with tarfile.open(fileobj=input_file, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise OSError("history archive is empty")
            for member in members:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise OSError("history archive contains an unsafe member path")
                if pure.parts[0] != expected_root:
                    raise OSError("history archive escaped its snapshot root")
                name = pure.as_posix().rstrip("/")
                if name in seen:
                    raise OSError("history archive contains duplicate member paths")
                seen.add(name)
                found_root = found_root or name == expected_root
                if member.isdir():
                    rows.append((name, "dir", 0, ""))
                elif member.issym():
                    _verify_link(pure, member.linkname, expected_root, hardlink=False)
                    rows.append((name, "link", 0, member.linkname))
                elif member.isfile() or member.islnk():
                    if member.islnk():
                        _verify_link(pure, member.linkname, expected_root, hardlink=True)
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise OSError("history archive payload cannot be read")
                    digest, size = hashlib.sha256(), 0
                    while block := payload.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                    rows.append((name, "file", size, digest.hexdigest()))
                else:
                    raise OSError("history archive contains an unsupported member type")
            if not found_root:
                raise OSError("history archive does not contain its snapshot root")
        current = path.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise OSError("history archive pathname changed during verification")
    return tuple(sorted(rows)), identity


def _verify_archive(path: Path, expected_root: str) -> tuple[tuple[str, str, int, str], ...]:
    return _verify_archive_with_identity(path, expected_root)[0]


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class HistoryRetention:
    """Compress and expire old backup snapshots using two size watermarks."""

    def __init__(self, history_dir: str, *, database_path: str, cache_dir: str,
                 compress_at_bytes: int, delete_at_bytes: int,
                 min_age_sec: float = 300, logger=None):
        requested = Path(history_dir).expanduser().absolute()
        if requested.exists() and requested.is_symlink():
            raise ValueError("storage.history_dir must not be a symlink")
        self.root = requested.resolve()
        self.database = Path(database_path).expanduser().absolute().resolve()
        self.cache = Path(cache_dir).expanduser().absolute().resolve()
        self.compress_at_bytes = int(compress_at_bytes)
        self.delete_at_bytes = int(delete_at_bytes)
        self.min_age_sec = max(0.0, float(min_age_sec))
        self.logger = logger
        self._validate_paths()
        if self.compress_at_bytes <= 0 or self.delete_at_bytes < self.compress_at_bytes:
            raise ValueError("history thresholds require 0 < compress <= delete")
        self._prepare_root()

    def _validate_paths(self) -> None:
        filesystem_root = Path(self.root.anchor).resolve()
        dangerous = {filesystem_root, Path.home().resolve(), self.database.parent, self.cache}
        if self.root in dangerous:
            raise ValueError("storage.history_dir is too broad or overlaps live Aurex storage")
        for protected in (self.database, self.cache):
            try:
                protected.relative_to(self.root)
            except ValueError:
                continue
            raise ValueError("storage.history_dir contains live Aurex storage")

    def _prepare_root(self) -> None:
        existed = self.root.exists()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("storage.history_dir must be a real directory")
        marker = self.root / _MARKER_NAME
        ordinary = [path for path in self.root.iterdir()
                    if path.name not in {_MARKER_NAME, _LOCK_NAME}
                    and not path.name.startswith(_TEMP_PREFIX)]
        trusted_legacy = self.root == (self.database.parent / "backups").resolve()
        if not marker.exists():
            if existed and ordinary and not trusted_legacy:
                raise ValueError(
                    "nonempty custom storage.history_dir lacks the Aurex history marker")
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(_MARKER_CONTENT)
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(self.root)
        if (marker.is_symlink() or not marker.is_file()
                or marker.stat().st_size != len(_MARKER_CONTENT)
                or marker.read_bytes() != _MARKER_CONTENT):
            raise ValueError("storage.history_dir has an invalid Aurex history marker")
        os.chmod(marker, 0o600)
        os.chmod(self.root, 0o700)

    @staticmethod
    def _is_archive(path: Path) -> bool:
        return path.name.endswith(_ARCHIVE_SUFFIX)

    def _entries(self) -> list[Path]:
        entries = []
        for path in self.root.iterdir():
            if (path.name in {_LOCK_NAME, _MARKER_NAME}
                    or path.name.startswith(_TEMP_PREFIX) or path.is_symlink()):
                continue
            if path.is_dir() or path.is_file():
                entries.append(path)
        return entries

    def usage_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(_tree_size(path) for path in self._entries())

    def _eligible(self, paths: list[Path], now: float) -> list[Path]:
        ages = [(path, _tree_mtime(path)) for path in paths]
        return sorted(
            (path for path, newest in ages if now - newest >= self.min_age_sec),
            key=lambda path: (_tree_mtime(path), path.name),
        )

    def _compress(self, source: Path) -> Path:
        if source.parent != self.root or source.is_symlink() or self._is_archive(source):
            raise ValueError("refusing to compress an unmanaged history path")
        if source.lstat().st_dev != self.root.lstat().st_dev or os.path.ismount(source):
            raise ValueError("refusing to compress a mounted history path")
        target = self.root / (source.name + _ARCHIVE_SUFFIX)
        if os.path.lexists(target):
            # Source+archive is an ambiguous crash/collision state.  Never
            # infer which copy is authoritative and never follow the target;
            # preserve both for explicit operator recovery.
            raise FileExistsError(
                f"history archive target already exists; preserving source: {target.name}")
        temporary = self.root / (_TEMP_PREFIX + uuid.uuid4().hex + _ARCHIVE_SUFFIX + ".partial")
        before = _snapshot(source)
        try:
            temporary.touch(mode=0o600, exist_ok=False)
            with tarfile.open(temporary, "w:gz", compresslevel=3, dereference=False) as archive:
                archive.add(source, arcname=source.name, recursive=True)
            archived = _verify_archive(temporary, source.name)
            # Verify the complete archive before the second stability check;
            # only an unchanged source is eligible for removal.
            if _snapshot(source) != before:
                raise OSError("history snapshot changed while it was being compressed")
            current = _source_signature(source)
            if _snapshot(source) != before or archived != current:
                raise OSError("history archive content does not match its source snapshot")
            with temporary.open("rb") as output:
                os.fchmod(output.fileno(), 0o600)
                os.utime(output.fileno(), ns=(time.time_ns(), max(row[3] for row in before)))
                os.fsync(output.fileno())
            # Link is no-clobber: a concurrent target collision preserves both
            # the source and our verified temporary archive.
            os.link(temporary, target, follow_symlinks=False)
            _fsync_directory(self.root)
            temporary.unlink()
            _fsync_directory(self.root)
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
            _fsync_directory(self.root)
            return target
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _delete(self, path: Path) -> None:
        if (path.parent != self.root or path.is_symlink() or not path.is_file()
                or not self._is_archive(path)):
            raise ValueError("refusing to delete an unverified history archive")
        _rows, identity = _verify_archive_with_identity(
            path, path.name[:-len(_ARCHIVE_SUFFIX)])
        current = path.lstat()
        if ((current.st_dev, current.st_ino) != identity
                or not stat_module.S_ISREG(current.st_mode)
                or current.st_nlink != 1):
            raise OSError("history archive changed before deletion")
        path.unlink()
        _fsync_directory(self.root)

    def enforce_once(self, *, max_operations: int | None = None,
                     stop_event: threading.Event | None = None) -> dict:
        """Apply retention once; failures preserve sources and are reported."""
        if (max_operations is not None and
                (type(max_operations) is not int or max_operations < 1)):
            raise ValueError("max_operations must be a positive integer or null")
        self._prepare_root()
        report = {
            "before_bytes": self.usage_bytes(), "after_bytes": 0,
            "compressed": [], "deleted": [], "partials_removed": [], "errors": [],
            "limited": False, "stopped": False,
        }
        lock_path = self.root / _LOCK_NAME
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = lock_path.lstat()
            if (not stat_module.S_ISREG(descriptor_stat.st_mode)
                    or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)):
                raise OSError("history retention lock must be a real regular file")
            lock = os.fdopen(descriptor, "a+b")
            descriptor = -1
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        with lock:
            os.fchmod(lock.fileno(), 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report["after_bytes"] = self.usage_bytes()
                report["locked"] = True
                return report
            # Exclusive ownership means no valid writer can own our temporary
            # files.  Clean crash remnants before measuring/rotating history.
            for partial in self.root.iterdir():
                if (partial.name.startswith(_TEMP_PREFIX) and partial.name.endswith(".partial")
                        and partial.is_file() and not partial.is_symlink()):
                    partial.unlink()
                    report["partials_removed"].append(partial.name)
            if report["partials_removed"]:
                _fsync_directory(self.root)
            now = time.time()
            operations = 0
            attempted_deletes: set[str] = set()

            def should_stop() -> bool:
                if stop_event is not None and stop_event.is_set():
                    report["stopped"] = True
                    return True
                if max_operations is not None and operations >= max_operations:
                    report["limited"] = True
                    return True
                return False

            def expire_verified_archives() -> None:
                nonlocal operations
                if self.usage_bytes() < self.delete_at_bytes:
                    return
                entries = self._entries()
                newest = max(entries, key=lambda path: (_tree_mtime(path), path.name), default=None)
                archives = self._eligible(
                    [path for path in entries
                     if self._is_archive(path) and path != newest
                     and path.name not in attempted_deletes], now)
                for archive in archives:
                    if self.usage_bytes() < self.delete_at_bytes or should_stop():
                        break
                    attempted_deletes.add(archive.name)
                    try:
                        self._delete(archive)
                        report["deleted"].append(archive.name)
                        operations += 1
                    except Exception as exc:
                        report["errors"].append({"entry": archive.name, "operation": "delete",
                                                  "error": str(exc)})

            # The delete watermark is the hard pressure path.  Prefer an
            # already verified old archive before spending this worker unit on
            # another compression (which briefly needs extra disk space).
            expire_verified_archives()

            if self.usage_bytes() >= self.compress_at_bytes:
                raw = self._eligible(
                    [path for path in self._entries() if not self._is_archive(path)], now)
                for source in raw:
                    if self.usage_bytes() < self.compress_at_bytes:
                        break
                    if should_stop():
                        break
                    try:
                        archive = self._compress(source)
                        report["compressed"].append({"source": source.name, "archive": archive.name})
                        operations += 1
                    except Exception as exc:
                        report["errors"].append({"entry": source.name, "operation": "compress",
                                                  "error": str(exc)})

            # Compression may itself cross the delete watermark.  In the
            # one-operation production worker this is reconsidered on the next
            # immediate pass; unlimited maintenance calls can converge now.
            expire_verified_archives()
            report["after_bytes"] = self.usage_bytes()
        return report


class HistoryRetentionWorker:
    """Run an immediate check and then periodic checks in one daemon thread."""

    def __init__(self, retention: HistoryRetention, interval_sec: float):
        self.retention = retention
        self.interval_sec = max(1.0, float(interval_sec))
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("history retention worker already started")
        self.thread = threading.Thread(target=self._loop, name="aurex-history-retention", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)

    def _loop(self) -> None:
        while not self.stopping.is_set():
            report = {}
            try:
                report = self.retention.enforce_once(
                    max_operations=1, stop_event=self.stopping)
                if self.retention.logger and (report["compressed"] or report["deleted"]
                                              or report["partials_removed"] or report["errors"]):
                    self.retention.logger.info("history retention: %s", report)
            except Exception as exc:
                if self.retention.logger:
                    self.retention.logger.error("history retention failed safely: %s", exc)
            delay = 0.1 if report.get("limited") and not report.get("errors") else self.interval_sec
            if self.stopping.wait(delay):
                break


__all__ = ["GIB", "HistoryRetention", "HistoryRetentionWorker"]
