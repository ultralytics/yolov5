from pathlib import Path
from preprocess_utils import find_missing_labels, copy_files_to_final_destination, subset_labels

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
label2id = {label: id for id, label in enumerate(label_lst)}


def main():
    # Create final destination for results if needed
    created_data_path = Path("created_data_table")
    (created_data_path / "images").mkdir(exist_ok=True, parents=True)
    (created_data_path / "labels").mkdir(exist_ok=True, parents=True)

    copy_files_to_final_destination(src_path='preprocess', dest_path='created_data_table')
    find_missing_labels('created_data_table/images', 'created_data_table/labels')
    subset_labels(labels_path='created_data_table/labels',
               subset=['Table', 'Table totals', 'Delivery address'],
               list_of_labels=label2id,
               merge_addresses=True)
    # TODO train split


if __name__ == "__main__":
    main()
