import os
import shutil

def organize_audio_by_prefix():
    # Start in current working directory
    base_dir = os.getcwd()

    # Target folders
    hfc_dir = os.path.join(base_dir, "HFC_audio")
    lfc_dir = os.path.join(base_dir, "LFC_audio")

    # Create subfolders
    os.makedirs(hfc_dir, exist_ok=True)
    os.makedirs(lfc_dir, exist_ok=True)

    # Audio file extensions to detect
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"}

    moved_hfc = 0
    moved_lfc = 0

    for root, _, files in os.walk(base_dir):
        # Skip target directories (to avoid moving files from output folders again)
        if root.startswith(hfc_dir) or root.startswith(lfc_dir):
            continue

        for file in files:
            # Check extension
            ext = os.path.splitext(file)[1].lower()
            if ext not in audio_extensions:
                continue

            # Full source file path
            source_path = os.path.join(root, file)

            # Check prefix
            if file.startswith("HFC"):
                target_folder = hfc_dir
                moved_hfc += 1
            elif file.startswith("LFC"):
                target_folder = lfc_dir
                moved_lfc += 1
            else:
                continue  # Skip files without HFC/LFC prefix

            # Target path
            target_path = os.path.join(target_folder, file)

            # Avoid duplicate names
            base, extension = os.path.splitext(target_path)
            counter = 1
            while os.path.exists(target_path):
                target_path = f"{base}_{counter}{extension}"
                counter += 1

            shutil.move(source_path, target_path)
            print(f"Moved: {target_path}")

    print(f"\nDone!")
    print(f"Moved {moved_hfc} HFC files.")
    print(f"Moved {moved_lfc} LFC files.")


if __name__ == "__main__":
    organize_audio_by_prefix()
