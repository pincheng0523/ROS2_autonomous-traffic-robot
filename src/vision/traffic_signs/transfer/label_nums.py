from pathlib import Path

labels_dir = Path("/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic_light/signs/transfer/TURN RIGHT TRAFFIC SIGN DATASET.v1-turn-right-traffic-sign-dataset.yolov8/test/labels")  
new_class_id = 5                                  

txt_files = list(labels_dir.glob("*.txt"))

if not txt_files:
    print(f"找不到任何 txt 檔：{labels_dir}")
    exit()

for txt_file in txt_files:
    new_lines = []

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            print(f"[警告] 格式異常，跳過：{txt_file.name} -> {line}")
            continue

        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"已修改：{txt_file.name}")

print("全部完成")
