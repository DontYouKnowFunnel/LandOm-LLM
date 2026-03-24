import hashlib
from typing import Iterable


def generate_hash(path: Iterable[int]) -> str:
    """DOM 순회 경로(path)로부터 항상 동일한 짧은 해시 ID를 생성합니다."""
    path_str = "-".join(map(str, path))
    return hashlib.md5(path_str.encode()).hexdigest()[:8]
