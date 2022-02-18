from pathlib import Path
from preprocess_utils import find_missing_labels, copy_files_to_final_destination, split_data

label_lst = [
    "Table",
    "Table head",
    "Table line",
    "Table column",
    "Table footer",
    "Table comments",
    "Table totals",
    "Delivery address",
    "Invoice address",
    "Vendor address",
    "Company address",
]

label2id = {
    "Table": 0,
    "Table head": None,
    "Table line": None,
    "Table column": None,
    "Table footer": None,
    "Table comments": None,
    "Table totals": 1,
    "Delivery address": 2,
    "Invoice address": 2,
    "Vendor address": 2,
    "Company address": 2,
}
label_pos = {label: id for id, label in enumerate(list(label2id.keys()))}


def main():
    # Create final destination for results if needed
    created_data_path = Path("created_data_rowcol")
    (created_data_path / "images").mkdir(exist_ok=True, parents=True)
    (created_data_path / "labels").mkdir(exist_ok=True, parents=True)

    copy_files_to_final_destination(src_path='preprocess', dest_path='created_data_rowcol')
    find_missing_labels('created_data_rowcol/images', 'created_data_rowcol/labels')
    split_data(labels_path='created_data_rowcol/labels',
               subset=['Table', 'Table head', 'Table line', 'Table column'],
               list_of_labels=label2id)
    # TODO get lines


if __name__ == "__main__":
    main()
