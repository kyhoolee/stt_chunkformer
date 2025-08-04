import os

def print_tree_summary(root_dir, max_files_preview=10, indent=""):
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        print(f"{indent}[Permission Denied]")
        return

    subdirs = [e for e in entries if os.path.isdir(os.path.join(root_dir, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root_dir, e))]

    # In thông tin folder hiện tại
    print(f"{indent}📁 {os.path.basename(root_dir)}/")

    # In subfolder
    for sub in subdirs:
        sub_path = os.path.join(root_dir, sub)
        print_tree_summary(sub_path, max_files_preview, indent + "    ")

    # In preview danh sách file
    if files:
        preview_files = files[:max_files_preview]
        for f in preview_files:
            print(f"{indent}    📄 {f}")
        if len(files) > max_files_preview:
            print(f"{indent}    ... ({len(files)} files total)")
        else:
            print(f"{indent}    ({len(files)} files total)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print folder tree with file preview")
    parser.add_argument("path", help="Root folder to scan")
    parser.add_argument("--max", type=int, default=10, help="Max files to preview per folder")
    args = parser.parse_args()

    print_tree_summary(args.path, max_files_preview=args.max)
