#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
YAML_FILE="/home/liutao/Project/GS-EVT/configs/VECTOR/sofa_normal1_config.yaml"
PYTHON_FILE="main.py"

max_events_per_frame=$(grep 'max_events_per_frame:' "$YAML_FILE" | awk '{print $2}')
save_path=$(grep 'save_path:' "$YAML_FILE" | awk '{print $2}')
new_max_events_per_frame=$max_events_per_frame

while [ "$new_max_events_per_frame" -ge 10000 ]; do
    new_save_path="${save_path}_max_events_per_frame${new_max_events_per_frame}"
    sed -i "s|save_path: .*|save_path: $new_save_path|" "$YAML_FILE"

    python3 "$PYTHON_FILE" "-c" "$YAML_FILE"

    new_max_events_per_frame=$((new_max_events_per_frame - 5000))
    sed -i "s/max_events_per_frame: .*/max_events_per_frame: $new_max_events_per_frame/" "$YAML_FILE"
    sed -i "s|save_path: .*|save_path: $save_path|" "$YAML_FILE"
done

sed -i "s/max_events_per_frame: .*/max_events_per_frame: $max_events_per_frame/" "$YAML_FILE"
sed -i "s|save_path: .*|save_path: $save_path|" "$YAML_FILE"