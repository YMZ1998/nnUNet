import os


def print_tree(start_path, prefix=""):
    items = sorted(os.listdir(start_path))[:]
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    folder = r"D:\Data\seg\open_atlas\test_atlas\data"
    print(f"\nDirectory tree for: {folder}\n")
    print_tree(folder)
