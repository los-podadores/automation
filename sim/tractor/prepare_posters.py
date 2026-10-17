"""Prepare photographic textures from the user's video and the local ROD test set.

This is asset extraction, with no generated imagery, annotations or YOLO boxes
painted onto the textures. Requires the existing Windows OpenCV environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/posters"


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Preparar fotos de los carteles")
    parser.add_argument(
        "--person-video", type=Path, default=Path.home() / "Videos/PODADORAS.mp4"
    )
    parser.add_argument("--person-image", type=Path)
    parser.add_argument("--second", type=float, default=80)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=(550, 80, 760, 478),
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument(
        "--dog-image",
        type=Path,
        default=ROOT.parents[1]
        / "datasets/rod/ROD-Dataset/dataset/test/images/IMG_20296.jpg",
    )
    args = parser.parse_args()
    if args.person_image:
        person = cv2.imread(str(args.person_image))
        source = dict(file=args.person_image.name, kind="user photograph")
    else:
        capture = cv2.VideoCapture(str(args.person_video))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, args.second * 1000)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            parser.error(f"No se pudo extraer el video: {args.person_video}")
        x1, y1, x2, y2 = args.crop
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            parser.error("Recorte fuera de las dimensiones del video")
        person = frame[y1:y2, x1:x2]
        source = dict(
            file=args.person_video.name,
            kind="user video",
            second=args.second,
            crop=list(args.crop),
        )
    if person is None or not args.dog_image.is_file():
        parser.error("Falta la foto de persona o perro")
    dog = cv2.imread(str(args.dog_image))
    if dog is None:
        parser.error("No se pudo leer la foto del perro")
    ASSETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ASSETS / "person.jpg"), person, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Focus the supplied ROD puppy photo on the animal; preserve its full height.
    # These coordinates are specific to this fixture, not a generated cutout.
    dog_crop = (
        (165, 0, 515, 640)
        if args.dog_image.name == "IMG_20296.jpg"
        else (0, 0, dog.shape[1], dog.shape[0])
    )
    x1, y1, x2, y2 = dog_crop
    cv2.imwrite(
        str(ASSETS / "dog.jpg"), dog[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    manifest = dict(
        person=source,
        dog=dict(
            file=args.dog_image.name,
            kind="local ROD dataset test image",
            citation="datasets/rod/citation_kaggle.md",
            crop=list(dog_crop),
        ),
        note="Local simulation fixtures, not a generalization benchmark. Retain source permissions before redistribution.",
        textures={
            name: dict(sha256=hashlib.sha256((ASSETS / name).read_bytes()).hexdigest())
            for name in ("person.jpg", "dog.jpg")
        },
    )
    (ASSETS / "sources.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fotos preparadas en {ASSETS}")


if __name__ == "__main__":
    main()
