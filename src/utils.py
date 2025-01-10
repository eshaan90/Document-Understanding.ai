import os
import json


def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def save_to_file(table_predictions, result_path, output_file_type='json'):
    if output_file_type == "json":
        with open(result_path, "w+", encoding="utf-8") as f:
            json.dump(table_predictions, f, ensure_ascii=False)