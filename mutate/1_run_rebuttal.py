import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import argparse

def count_total_rows(csv_files):
    """
    Count the total number of rows across multiple CSV files.
    :param csv_files: list of CSV file paths
    :return: total rows across all CSV files
    """
    total_rows = 0
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_rows

def run_scripts_in_directory(directory, script_sequence, script_args_map, script_csv_map):
    """
    Run scripts in a single directory in a specific order.
    
    :param directory: path to the directory
    :param script_sequence: list of (script_name, auto_repeat)
    :param script_args_map: dict => script_name -> [arg1, arg2, ...] (if needed)
    :param script_csv_map: dict => script_name -> list_of_csv_files
    """
    print(f"[{directory}] Starting scripts...")

    for script_name, auto_repeat in script_sequence:
        script_path = os.path.join(directory, script_name)
        # get args for this script
        args = script_args_map.get(script_name, [])

        # which csv files does this script check? if not specified, default to None
        csv_files_for_script = script_csv_map.get(script_name, [])

        while True:
            print(f"[{directory}] Running {script_name} with args: {args}")
            try:
                subprocess.run(["python", script_path] + args, check=True)
                print(f"[{directory}] Finished {script_name}")
            except subprocess.CalledProcessError as e:
                print(f"[{directory}] Error running {script_name}: {e}")
                break

            if auto_repeat:
                # 如果需要自动重复，检查对应CSV的行数
                if csv_files_for_script:
                    total_rows = count_total_rows(csv_files_for_script)
                    print(f"[{directory}] {script_name} row check: total_rows={total_rows}")
                    if total_rows <= 60:
                        print(f"[{directory}] {script_name}: Rows <= 60, repeating...")
                    else:
                        print(f"[{directory}] {script_name}: Rows > 60, move on.")
                        break
                else:
                    print(f"[{directory}] {script_name}: no CSV set, no check performed, proceed next.")
                    break
            else:
                # 不需要重复的脚本，直接执行一次
                break
    
    print(f"[{directory}] All scripts done in this directory.")

def run_all_directories(directories, script_sequence, script_args_map, script_csv_map):
    """
    Run scripts for each directory in parallel.
    :param directories: list of directory paths
    :param script_sequence: e.g. [("scriptA.py", True), ("scriptB.py", False), ...]
    :param script_args_map: dict => { script_name: [args] }
    :param script_csv_map: dict => { script_name: [csv1, csv2, ...] } for row-check
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for d in directories:
            futures.append(
                executor.submit(run_scripts_in_directory, d, script_sequence, script_args_map, script_csv_map)
            )
        for f in futures:
            f.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()

    # 1. 定义 6 个目录
    directories = [
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_1",
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_2",
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_3",
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_4",
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_5",
        f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations/{args.book}/original_question_6"
    ]

    # 2. 修改脚本执行顺序，只运行脚本 2, 4, 6, 8
    script_sequence = [
        ("2_eval_zero_shot_with_judge.py", False), 
        ("4_eval_zero_shot_without_judge.py", False),
        ("6_eval_few_shots_with_judge.py", False),
        ("8_eval_few_shots_without_judge.py", False),
    ]

    # 3. 脚本参数映射（保持不变）
    script_args_map = {
        "1_zero_shot_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "2_eval_zero_shot_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "3_zero_shot_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "4_eval_zero_shot_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "5_few_shots_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "6_eval_few_shots_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "7_few_shots_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "8_eval_few_shots_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
    }

    # 4. 脚本 CSV 映射（保持不变，但由于我们不重复执行脚本1,3,5,7，此处不会被用到）
    script_csv_map = {
        "1_zero_shot_with_judge.py": [
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/1_zero_shot_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/2_zero_shot_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/3_zero_shot_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/4_zero_shot_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/5_zero_shot_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/6_zero_shot_with_judge.csv"
        ],
        "5_few_shots_with_judge.py": [
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/1_few_shots_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/2_few_shots_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/3_few_shots_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/4_few_shots_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/5_few_shots_with_judge.csv",
            f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/2_persuasion_prompts/{args.book}/{args.technique_dir}/6_few_shots_with_judge.csv"
        ]
    }

    # 5. 开始执行
    run_all_directories(directories, script_sequence, script_args_map, script_csv_map)
    print("All directories completed.")

"""
执行示例：
python run.py --book 1_HP --technique "Foot-in-the-Door" --technique_dir 16_Foot-in-the-Door
"""