import os
import shutil
import gradio as gr


def Upload_files(files, progress=gr.Progress()):
    op = ""
    progress(0.1, desc="Uploading the files...")
    for file in files:
        print(f"uploading the file {file}")
        file_suffix = file.name.split(".")[-1]
        file_path = "./assets/doc_list"
        if str(file_suffix).lower() == "pdf":
            file_path = os.path.join(file_path, "pdf_files")
        elif str(file_suffix).lower() == "html":
            file_path = os.path.join(file_path, "html_files")
        elif str(file_suffix).lower() == "txt":
            file_path = os.path.join(file_path, "text_files")
        else:
            print(f"invalid file suffix {str(file_suffix).lower()}")
            return

        os.makedirs(file_path, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(file_path, os.path.basename(file.name))
        copy_file(file.name, file_path)
    op = f"successfully copied {len(files)} files"
    print(op)
    progress(0.2, desc=op)


def copy_file(source_file, destination_file):
    print(f"copying {source_file} to {destination_file}")
    try:
        shutil.copy2(source_file, destination_file)
    except:
        print(f"error failed to copy {source_file} to {destination_file}")
