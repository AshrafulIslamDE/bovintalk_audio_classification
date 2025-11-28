import os
import shutil

def organize_audio_files():
    # Current working directory
    source_dir = os.getcwd()
    target_dir = os.path.join(source_dir, "organized_audio")

    # Audio extensions to collect
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"}

    # Create target folder
    os.makedirs(target_dir, exist_ok=True)

    moved_count = 0

    for root, _, files in os.walk(source_dir):
        # Skip the target directory itself (avoid infinite loop)
        if root.startswith(target_dir):
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                source_path = os.path.join(root, file)

                # Prepare target file path (avoid duplication)
                dest_path = os.path.join(target_dir, file)
                base, extension = os.path.splitext(dest_path)

                counter = 1
                while os.path.exists(dest_path):
                    dest_path = f"{base}_{counter}{extension}"
                    counter += 1

                # Move file
                shutil.move(source_path, dest_path)
                moved_count += 1
                print(f"Moved: {dest_path}")

    print(f"\nDone! Total audio files moved: {moved_count}")


if __name__ == "__main__":
    organize_audio_files()
