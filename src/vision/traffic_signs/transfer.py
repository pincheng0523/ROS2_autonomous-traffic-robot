import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_box(size_w, size_h, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2.0) / size_w
    y_center = ((ymin + ymax) / 2.0) / size_h
    box_w = (xmax - xmin) / size_w
    box_h = (ymax - ymin) / size_h
    return x_center, y_center, box_w, box_h


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    if size is None:
        raise ValueError(f"XML 缺少 <size>: {xml_path}")

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            print(f"[警告] 跳過非法框: {xml_path}")
            continue

        objects.append({
            "class_name": class_name,
            "bbox": (xmin, ymin, xmax, ymax)
        })

    return filename, width, height, objects


def find_image_file(images_dir, filename_from_xml):
    if filename_from_xml:
        candidate = images_dir / filename_from_xml
        if candidate.exists():
            return candidate

    stem = Path(filename_from_xml).stem if filename_from_xml else None
    if stem:
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    # ======== 你只要改這裡 ========
    images_dir = Path("/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic light/signs/images")      # 原始圖片資料夾
    annotations_dir = Path("/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic light/signs/annotations")   # 原始 XML 資料夾
    output_dir = Path("/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic light/signs/transfer")        # 輸出資料夾
    train_ratio = 0.8
    random_seed = 42
    # =============================

    random.seed(random_seed)

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到圖片資料夾: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"找不到 XML 資料夾: {annotations_dir}")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"在 {annotations_dir} 找不到任何 XML")

    all_samples = []
    class_names = set()

    print("開始讀取 XML...")
    for xml_file in xml_files:
        try:
            filename, width, height, objects = parse_voc_xml(xml_file)
        except Exception as e:
            print(f"[錯誤] 解析失敗 {xml_file.name}: {e}")
            continue

        image_path = find_image_file(images_dir, filename)
        if image_path is None:
            print(f"[警告] 找不到對應圖片: XML={xml_file.name}, filename={filename}")
            continue

        if not objects:
            print(f"[警告] 沒有標註物件，仍保留圖片但標籤為空: {xml_file.name}")

        for obj in objects:
            class_names.add(obj["class_name"])

        all_samples.append({
            "xml_path": xml_file,
            "image_path": image_path,
            "width": width,
            "height": height,
            "objects": objects,
        })

    if not all_samples:
        raise RuntimeError("沒有可用資料，請檢查 XML 與圖片對應是否正確")

    class_names = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    print("類別如下:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    random.shuffle(all_samples)
    split_index = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_index]
    val_samples = all_samples[split_index:]

    if len(val_samples) == 0:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]

    # 建立輸出資料夾
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        ensure_dir(p)

    def process_split(samples, img_out_dir, lbl_out_dir, split_name):
        print(f"處理 {split_name}: {len(samples)} 筆")
        for sample in samples:
            image_path = sample["image_path"]
            width = sample["width"]
            height = sample["height"]
            objects = sample["objects"]

            # 複製圖片
            dst_img = img_out_dir / image_path.name
            shutil.copy2(image_path, dst_img)

            # 建立對應 label txt
            label_path = lbl_out_dir / f"{image_path.stem}.txt"
            lines = []

            for obj in objects:
                class_id = class_to_id[obj["class_name"]]
                xmin, ymin, xmax, ymax = obj["bbox"]
                x_center, y_center, box_w, box_h = convert_box(
                    width, height, xmin, ymin, xmax, ymax
                )
                lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                )

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    process_split(train_samples, train_img_dir, train_lbl_dir, "train")
    process_split(val_samples, val_img_dir, val_lbl_dir, "val")

    # classes.txt
    classes_txt = output_dir / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    # data.yaml
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print("\n完成")
    print(f"輸出資料夾: {output_dir}")
    print(f"classes.txt: {classes_txt}")
    print(f"data.yaml: {data_yaml}")
    print(f"train 數量: {len(train_samples)}")
    print(f"val 數量: {len(val_samples)}")


if __name__ == "__main__":
    main()
