from pathlib import Path
from types import SimpleNamespace
import os, yaml, re, copy

# Regex for ${...} placeholders
_VAR = re.compile(r"\$\{([^}]+)\}")

def _lookup_ctx(ctx: dict, dotted: str):
    cur = ctx
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Missing key '{part}' while resolving '{dotted}'")
    return cur

def _resolve_placeholders(text: str, ctx: dict) -> str:
    def repl(m):
        key = m.group(1).strip()
        if key.lower().startswith("env:"):
            return os.environ.get(key[4:], "")
        if key.lower().startswith("env."):
            return os.environ.get(key[4:], "")
        if key in os.environ:
            return os.environ[key]
        return str(_lookup_ctx(ctx, key))
    return _VAR.sub(repl, text)

def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj

def _expand_path(p: str) -> str:
    return str(Path(os.path.expanduser(os.path.expandvars(p))))

def _expand_paths_inplace(obj):
    # Recursively expand every string value
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = _expand_path(v)
            else:
                _expand_paths_inplace(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = _expand_path(v)
            else:
                _expand_paths_inplace(v)

def _dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(x) for x in d]
    else:
        return d

def load_cfg(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

  
    resolved = copy.deepcopy(raw)
    for _ in range(5):
        dumped = yaml.safe_dump(resolved, sort_keys=False)
        if not _VAR.search(dumped):
            break
        dumped = _resolve_placeholders(dumped, resolved)
        resolved = yaml.safe_load(dumped) or {}

   
    if isinstance(resolved.get("paths"), dict):
        _expand_paths_inplace(resolved["paths"])

    
    return _dict_to_namespace(_normalize(resolved))
