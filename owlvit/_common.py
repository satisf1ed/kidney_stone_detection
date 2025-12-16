from pathlib import Path
import yaml

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def resolve(path, base: Path):
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

def load_prompts(prompts_yaml: Path):
    d = yaml.safe_load(prompts_yaml.read_text(encoding="utf-8"))
    prompts = d.get("prompts") or ["kidney stone"]
    return [str(x) for x in prompts]

def get_images_dir_from_data_yaml(data_yaml: Path, split: str):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    key = split
    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        key = "valid"
    sp = resolve(d.get(key), base)
    if sp.name != "images" and (sp / "images").exists():
        sp = sp / "images"
    return sp
