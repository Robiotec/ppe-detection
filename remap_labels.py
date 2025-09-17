import os
from glob import glob

# Mapeo original -> nuevo
MAP = {
    0: 0,  # person
    10: 1,  # helmet
    16: 2,  # safety-vest
    8: 3,  # glasses
    9: 4,  # gloves
    5: 5,  # face-mask
    14: 6,  # shoes
}


def process_label_file(src_file, dst_file):
    new_lines = []
    with open(src_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls in MAP:
                new_cls = MAP[cls]
                new_line = " ".join([str(new_cls)] + parts[1:])
                new_lines.append(new_line)

    if new_lines:
        with open(dst_file, "w") as f:
            f.write("\n".join(new_lines))


def remap_split(split):
    src_dir = f"{split}/labels"
    dst_dir = f"{split}/labels_mapped"
    os.makedirs(dst_dir, exist_ok=True)

    for file in glob(os.path.join(src_dir, "*.txt")):
        fname = os.path.basename(file)
        dst_file = os.path.join(dst_dir, fname)
        process_label_file(file, dst_file)


if __name__ == "__main__":
    for split in ["train", "val"]:
        remap_split(split)
    print(
        "✅ Remapeo terminado. Archivos guardados en train/labels_mapped y val/labels_mapped"
    )
