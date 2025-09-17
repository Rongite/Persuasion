import pandas as pd


'''
modify # every 6 questions
'''
df = pd.read_csv('/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/1_HP/16_Foot-in-the-Door/5_few_shot_version_single_evaluation_results_continue_other_HP_500.csv')

# Output the valid mutations DataFrame
print(df)

# Calculate averages for the model 'claude-3-haiku-20240307'
claude_avg = df[df['model'] == 'claude-3-haiku-20240307'][['rouge1_precision', 'rougeL_precision']].mean()
print("Claude-3-Haiku-20240307 Average Scores:")
print(claude_avg)

# Calculate averages for the model 'gpt-4o-mini'
gpt_avg = df[df['model'] == 'gpt-4o-mini'][['rouge1_precision', 'rougeL_precision']].mean()
print("GPT-4o-Mini Average Scores:")
print(gpt_avg)
