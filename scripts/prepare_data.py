import sys
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError, ImageFile
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    DATA_DIR,
    RAW_DIR,
    MIN_SIZE,
    MAX_ASPECT_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    SEED,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoSubsetDownloader:
    def __init__(
        self,
        ann_file: str | Path,
        out_dir: str | Path,
        n_images: int = 10_000,
        min_size: int = 256,
        seed: int = 42,
        timeout: int = 20,
        retries: int = 3,
        num_threads: int = 10,
    ) -> None:
        self.ann_file = Path(ann_file)
        self.out_dir = Path(out_dir)
        self.n_images = n_images
        self.min_size = min_size
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.num_threads = num_threads

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.coco = COCO(str(self.ann_file))
        self.rng = random.Random(seed)

    def _valid_img_infos(self) -> list[dict]:
        img_ids = self.coco.getImgIds()
        infos = self.coco.loadImgs(img_ids)

        valid = []
        for info in infos:
            width = info.get("width", 0)
            height = info.get("height", 0)
            coco_url = info.get("coco_url", "")
            file_name = info.get("file_name", "")

            if not coco_url or not file_name:
                continue
            if width < self.min_size or height < self.min_size:
                continue

            valid.append(info)

        return valid

    def _download_one(self, info: dict) -> tuple[str, bool, str]:
        file_name = info["file_name"]
        url = info["coco_url"]
        dst_path = self.out_dir / file_name

        if dst_path.exists():
            return file_name, True, "already exists"

        for attempt in range(1, self.retries + 1):
            try:
                with requests.get(url, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    with open(dst_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                with Image.open(dst_path) as img:
                    img = img.convert("RGB")
                    _ = img.size

                return file_name, True, "downloaded"

            except Exception as e:
                if dst_path.exists():
                    dst_path.unlink(missing_ok=True)
                if attempt < self.retries:
                    time.sleep(1.0)
                else:
                    return file_name, False, str(e)

        return file_name, False, "unknown error"

    def download(self) -> None:
        infos = self._valid_img_infos()
        if len(infos) < self.n_images:
            raise ValueError(
                f"Requested {self.n_images} images, but only {len(infos)} valid images found."
            )

        sampled = self.rng.sample(infos, self.n_images)

        success_count = 0
        failed = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._download_one, info) for info in sampled]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading COCO"):
                file_name, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    failed.append((file_name, msg))

        print(f"Downloaded successfully: {success_count}/{self.n_images}")

        if failed:
            print(f"Failed: {len(failed)} files")
            for file_name, msg in failed[:10]:
                print(f"[FAILED] {file_name}: {msg}")


def is_valid_image(path: str | Path, min_size: int = 128, max_aspect_ratio: float = 3.0) -> bool:
    path = Path(path)

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w < min_size or h < min_size:
                return False

            ratio = max(w / h, h / w)
            if ratio > max_aspect_ratio:
                return False

        return True

    except (UnidentifiedImageError, OSError, ValueError):
        return False


def collect_valid_images(
    image_dir: str | Path,
    valid_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    min_size: int = 128,
    max_aspect_ratio: float = 3.0,
) -> list[Path]:
    image_dir = Path(image_dir)

    all_paths = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in valid_exts
    ]

    valid_paths = [
        p for p in all_paths
        if is_valid_image(p, min_size=min_size, max_aspect_ratio=max_aspect_ratio)
    ]

    return sorted(valid_paths)


def save_split(name: str, paths: list[Path], splits_dir: Path) -> None:
    file_path = splits_dir / f"{name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([str(p) for p in paths], f, indent=2, ensure_ascii=False)
    print(f"{name}: {len(paths)} -> {file_path}")


def main():
    ann_file = DATA_DIR / "annotations" / "instances_train2017.json"
    raw_dir = RAW_DIR
    splits_dir = DATA_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    downloader = CocoSubsetDownloader(
        ann_file=ann_file,
        out_dir=raw_dir,
        n_images=10_000,
        min_size=256,
        seed=SEED,
        num_threads=30,
    )
    downloader.download()

    valid_images = collect_valid_images(
        raw_dir,
        min_size=MIN_SIZE,
        max_aspect_ratio=MAX_ASPECT_RATIO,
    )

    print(f"Valid images found after filtering: {len(valid_images)}")

    random.seed(SEED)
    random.shuffle(valid_images)

    total = len(valid_images)
    train_len = int(total * TRAIN_RATIO)
    val_len = int(total * VAL_RATIO)

    train_paths = valid_images[:train_len]
    val_paths = valid_images[train_len:train_len + val_len]
    test_paths = valid_images[train_len + val_len:]

    save_split("train", train_paths, splits_dir)
    save_split("val", val_paths, splits_dir)
    save_split("test", test_paths, splits_dir)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
