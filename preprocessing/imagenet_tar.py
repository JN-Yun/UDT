import os
import sys
import tarfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_class_tar(args):
    # Unpack filename and root directory from the argument tuple
    tar_filename, root_dir = args
    if not tar_filename.endswith(".tar"):
        return

    class_tar_path = os.path.join(root_dir, tar_filename)
    class_name = tar_filename[:-4]
    class_dir = os.path.join(root_dir, class_name)

    os.makedirs(class_dir, exist_ok=True)

    try:
        with tarfile.open(class_tar_path) as class_tar:
            class_tar.extractall(path=class_dir)
        os.remove(class_tar_path)
    except Exception as e:
        print(f"Error extracting {tar_filename}: {e}")

if __name__ == "__main__":
    tar_path = "./ILSVRC2012_img_train.tar"
    extract_root = "./train"

    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            value = value.strip('"\'')
            
            if key == "tar_path":
                tar_path = value
            elif key == "extract_root":
                extract_root = value

    print(f"Target TAR: {tar_path}")
    print(f"Extract Root: {extract_root}")

    # Check if the tar file exists
    if not os.path.exists(tar_path):
        print(f"Error: Could not find the file {tar_path}.")
        sys.exit(1)

    os.makedirs(extract_root, exist_ok=True)

    print("Extracting the main tar file...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_root)

    tar_files = [
        f for f in os.listdir(extract_root)
        if f.endswith(".tar")
    ]

    tasks = [(f, extract_root) for f in tar_files]

    num_workers = min(cpu_count(), 16) 
    print(f"Using {num_workers} workers")

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(extract_class_tar, tasks),
            total=len(tasks)
        ))