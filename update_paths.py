#!/usr/bin/env python3
import os
import re

def update_paths_in_file(file_path):
    """Update absolute paths to relative paths in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: Replace /home/jlong1/Downloads/persuasion/Data_n_Code_persuasion with current directory
        content = re.sub(
            r'/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion',
            '.',
            content
        )

        # Pattern 2: Handle specific outputs paths
        content = re.sub(
            r'/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/outputs',
            './outputs',
            content
        )

        # Pattern 3: Handle mutate paths
        content = re.sub(
            r'/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/mutate',
            './mutate',
            content
        )

        # Write back if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_all_python_files(root_dir):
    """Update all Python files in the directory tree"""
    updated_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_paths_in_file(file_path):
                    updated_count += 1

    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    # Update all Python files in the current directory and subdirectories
    update_all_python_files('/home/jlong1/Downloads/Persuasion/mutate')