"""样本推理的模型输入缓存。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from ramanv2.common.naming import build_natural_key
from ramanv2.common.publishing import create_temporary_path, publish_directory
from ramanv2.inference.directory import list_spectrum_paths
from ramanv2.inference.predictor import Predictor
from ramanv2.inference.spectra import build_inference_preprocessor, preprocess_spectrum_path


MANIFEST_NAME = "cache_manifest.json"
CACHE_VERSION = 1


@dataclass(frozen=True)
class CachedSpectrum:
    file_name: str
    path: Path


@dataclass(frozen=True)
class CachedFolder:
    folder_name: str
    files: tuple[CachedSpectrum, ...]


def build_test_cache(
    source_root: Path | str,
    cache_root: Path | str,
    folder_names: list[str] | tuple[str, ...] | set[str],
    predictor: Predictor,
) -> dict[str, object]:
    """增量构建已经完成模型输入预处理的测试光谱缓存。"""
    source = Path(source_root).resolve()
    cache = Path(cache_root).resolve()
    selected = sorted({str(name) for name in folder_names}, key=build_natural_key)
    if not selected:
        raise ValueError("至少选择一个 CS 文件夹")
    if not source.is_dir():
        raise FileNotFoundError(f"CSdata 不存在：{source}")

    old_manifest = _read_manifest(cache / MANIFEST_NAME)
    input_signature = _input_signature(predictor)
    entries = dict(old_manifest.get("folders", {}))
    temporary = create_temporary_path(cache)
    reused_count = 0
    rebuilt_count = 0
    file_count = 0
    failures: list[dict[str, str]] = []
    preprocessor = build_inference_preprocessor(predictor.input_spec, predictor.device)
    try:
        if cache.is_dir():
            shutil.copytree(cache, temporary, dirs_exist_ok=True)
        else:
            temporary.mkdir(parents=True, exist_ok=False)
        for folder_name in selected:
            source_folder = source / folder_name
            if not source_folder.is_dir():
                raise FileNotFoundError(f"CS 文件夹不存在：{source_folder}")
            spectrum_paths = list_spectrum_paths(source_folder)
            if not spectrum_paths:
                raise ValueError(f"CS 文件夹没有 .arc_data：{source_folder}")
            old_files = entries.get(folder_name, {}).get("files", {})
            if not isinstance(old_files, dict):
                old_files = {}
            target_folder = temporary / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)
            new_files: dict[str, dict[str, object]] = {}
            for source_path in spectrum_paths:
                source_info = _source_info(source_path)
                old_info = old_files.get(source_path.name)
                target_path = target_folder / f"{source_path.stem}.pt"
                if (
                    isinstance(old_info, dict)
                    and old_info.get("source") == source_info
                    and old_info.get("input_signature") == input_signature
                    and target_path.is_file()
                ):
                    reused_count += 1
                else:
                    try:
                        tensor = preprocess_spectrum_path(
                            source_path,
                            preprocessor,
                            predictor.input_config.bad_bands,
                        ).detach().cpu()
                        torch.save(
                            {
                                "input": tensor,
                                "file_name": source_path.name,
                                "source_path": str(source_path),
                                "profile_id": predictor.profile_id,
                                "input_signature": input_signature,
                            },
                            target_path,
                        )
                        rebuilt_count += 1
                    except (OSError, RuntimeError, ValueError) as error:
                        failures.append({"file": str(source_path), "error": str(error)})
                        continue
                new_files[source_path.name] = {
                    "cache_path": target_path.relative_to(temporary).as_posix(),
                    "source": source_info,
                    "input_signature": input_signature,
                }
                file_count += 1
            kept_cache_names = {
                Path(str(file_entry["cache_path"])).name
                for file_entry in new_files.values()
            }
            for stale_path in target_folder.glob("*.pt"):
                if stale_path.name not in kept_cache_names:
                    stale_path.unlink()
            if failures and not new_files:
                raise RuntimeError(f"文件夹缓存构建失败：{folder_name}")
            entries[folder_name] = {
                "file_count": len(new_files),
                "files": new_files,
            }
        manifest = {
            "version": CACHE_VERSION,
            "profile_id": predictor.profile_id,
            "input_config": _json_ready(asdict(predictor.input_config)),
            "wavenumber_range": [
                float(predictor.input_config.cut_min),
                float(predictor.input_config.cut_max),
            ],
            "target_points": int(predictor.input_config.target_points),
            "bad_bands": _json_ready(predictor.input_config.bad_bands),
            "input_signature": input_signature,
            "folders": entries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if failures:
            raise RuntimeError("部分光谱预处理失败，缓存未发布：" + str(failures))
        publish_directory(temporary, cache)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "ready",
        "cache_root": str(cache),
        "profile_id": predictor.profile_id,
        "folder_names": selected,
        "file_count": file_count,
        "reused_count": reused_count,
        "rebuilt_count": rebuilt_count,
        "failed_count": len(failures),
        "failures": failures,
    }


def load_test_cache(
    cache_root: Path | str,
    folder_names: list[str] | tuple[str, ...] | set[str],
    predictor: Predictor,
) -> list[CachedFolder]:
    """读取并校验指定文件夹的模型输入缓存。"""
    cache = Path(cache_root).resolve()
    manifest = _read_manifest(cache / MANIFEST_NAME)
    expected_signature = _input_signature(predictor)
    if manifest.get("profile_id") != predictor.profile_id:
        raise ValueError("测试缓存 Profile 与实验不一致")
    folders = manifest.get("folders", {})
    if not isinstance(folders, dict):
        raise ValueError("测试缓存目录清单无效")
    result: list[CachedFolder] = []
    for folder_name in sorted({str(name) for name in folder_names}, key=build_natural_key):
        folder_entry = folders.get(folder_name)
        if not isinstance(folder_entry, dict):
            raise FileNotFoundError(f"缺少测试缓存文件夹：{folder_name}")
        files = folder_entry.get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"测试缓存文件清单无效：{folder_name}")
        cached_files: list[CachedSpectrum] = []
        for file_name, file_entry in sorted(files.items(), key=lambda item: build_natural_key(item[0])):
            if not isinstance(file_entry, dict) or file_entry.get("input_signature") != expected_signature:
                raise ValueError(f"测试缓存输入规格已变化，请重新构建：{folder_name}/{file_name}")
            path = cache / str(file_entry.get("cache_path", ""))
            if not path.is_file():
                raise FileNotFoundError(f"测试缓存文件不存在：{path}")
            cached_files.append(CachedSpectrum(str(file_name), path))
        if not cached_files:
            raise ValueError(f"测试缓存文件夹为空：{folder_name}")
        result.append(CachedFolder(folder_name, tuple(cached_files)))
    if not result:
        raise ValueError("没有选择测试缓存文件夹")
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": CACHE_VERSION, "folders": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "folders": {}}
    return payload


def _source_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _input_signature(predictor: Predictor) -> str:
    payload = {
        "profile_id": predictor.profile_id,
        "input_config": _json_ready(asdict(predictor.input_config)),
        "input_spec": _json_ready(asdict(predictor.input_spec)),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


__all__ = ["CachedFolder", "CachedSpectrum", "build_test_cache", "load_test_cache"]
