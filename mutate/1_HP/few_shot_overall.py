# # few shots Foot-in-the-Door 
# claude_3_haiku_mutation_avg_rouge1 = (0.194624 + 0.216868 + 0.269237 + 0.187266 + 0.188662 + 0.171718) / 6
# print("claude_3_haiku_mutation_avg_rouge1: ", claude_3_haiku_mutation_avg_rouge1, "\n")
# claude_3_haiku_mutation_avg_rougeL = (0.133803 + 0.127158 + 0.161554 + 0.096808 + 0.099505 + 0.093345) / 6
# print("claude_3_haiku_mutation_avg_rougeL: ", claude_3_haiku_mutation_avg_rougeL, "\n")
# gpt_4o_mini_mutation_avg_rouge1 = (0.274618 + 0.252676 + 0.267879 + 0.263918 + 0.255323 + 0.398201) / 6
# print("gpt_4o_mini_mutation_avg_rouge1: ", gpt_4o_mini_mutation_avg_rouge1, "\n")
# gpt_4o_mini_mutation_avg_rougeL = (0.148803 + 0.127668 + 0.142529 + 0.128218 + 0.130330 + 0.332895) / 6
# print("gpt_4o_mini_mutation_avg_rougeL: ", gpt_4o_mini_mutation_avg_rougeL, "\n")
# valid_mutation_rate = (93 + 74 + 98 + 81 + 62 + 59) / (100 + 100 + 100 + 100 + 100 + 100)
# print("valid_mutation_rate: ", valid_mutation_rate)

# # few shots Storytelling
# claude_3_haiku_mutation_avg_rouge1 = (0.166017 + 0.152678 + 0.253520 + 0.179536 + 0.187506 + 0.194885) / 6
# print("claude_3_haiku_mutation_avg_rouge1: ", claude_3_haiku_mutation_avg_rouge1, "\n")
# claude_3_haiku_mutation_avg_rougeL = (0.084078 + 0.079794 + 0.143863 + 0.094261 + 0.095821 + 0.129270) / 6
# print("claude_3_haiku_mutation_avg_rougeL: ", claude_3_haiku_mutation_avg_rougeL, "\n")
# gpt_4o_mini_mutation_avg_rouge1 = (0.304258 + 0.274168 + 0.293692 + 0.290520 + 0.283977 + 0.271432) / 6
# print("gpt_4o_mini_mutation_avg_rouge1: ", gpt_4o_mini_mutation_avg_rouge1, "\n")
# gpt_4o_mini_mutation_avg_rougeL = (0.162606 + 0.138878 + 0.197412 + 0.144622 + 0.151778 + 0.142035) / 6
# print("gpt_4o_mini_mutation_avg_rougeL: ", gpt_4o_mini_mutation_avg_rougeL, "\n")
# valid_mutation_rate = (63 + 23 + 93 + 48 + 81 + 74) / (100 + 100 + 100 + 100 + 100 + 100)
# print("valid_mutation_rate: ", valid_mutation_rate)

# # few shots Encouragement
# claude_3_haiku_mutation_avg_rouge1 = (0.181939 + 0.234979 + 0.205122 + 0.194677 + 0.193403 + 0.209367) / 6
# print("claude_3_haiku_mutation_avg_rouge1: ", claude_3_haiku_mutation_avg_rouge1, "\n")
# claude_3_haiku_mutation_avg_rougeL = (0.091259 + 0.159315 + 0.094957 + 0.095783 + 0.093720 + 0.104760) / 6
# print("claude_3_haiku_mutation_avg_rougeL: ", claude_3_haiku_mutation_avg_rougeL, "\n")
# gpt_4o_mini_mutation_avg_rouge1 = (0.283311 + 0.285956 + 0.265426 + 0.278936 + 0.284036 + 0.280353) / 6
# print("gpt_4o_mini_mutation_avg_rouge1: ", gpt_4o_mini_mutation_avg_rouge1, "\n")
# gpt_4o_mini_mutation_avg_rougeL = (0.148957 + 0.147296 + 0.139234 + 0.147119 + 0.149633 + 0.147781) / 6
# print("gpt_4o_mini_mutation_avg_rougeL: ", gpt_4o_mini_mutation_avg_rougeL, "\n")
# valid_mutation_rate = (42 + 44 + 40 + 71 + 95 + 76) / (100 + 100 + 100 + 100 + 100 + 100)
# print("valid_mutation_rate: ", valid_mutation_rate)

# few shots Negotiation
claude_3_haiku_mutation_avg_rouge1 = (0.185177 + 0.159967 + 0.210304 + 0.179732 + 0.183720 + 0.187477) / 6
print("claude_3_haiku_mutation_avg_rouge1: ", claude_3_haiku_mutation_avg_rouge1, "\n")
claude_3_haiku_mutation_avg_rougeL = (0.091021 + 0.076720 + 0.105979 + 0.087927 + 0.086478 + 0.090985) / 6
print("claude_3_haiku_mutation_avg_rougeL: ", claude_3_haiku_mutation_avg_rougeL, "\n")
gpt_4o_mini_mutation_avg_rouge1 = (0.257322 + 0.256130 + 0.269065 + 0.259014 + 0.256608 + 0.265079) / 6
print("gpt_4o_mini_mutation_avg_rouge1: ", gpt_4o_mini_mutation_avg_rouge1, "\n")
gpt_4o_mini_mutation_avg_rougeL = (0.131083 + 0.128268 + 0.138850 + 0.130438 + 0.127918 + 0.137070) / 6
print("gpt_4o_mini_mutation_avg_rougeL: ", gpt_4o_mini_mutation_avg_rougeL, "\n")
valid_mutation_rate = (75 + 81 + 52 + 43 + 62 + 31) / (100 + 100 + 100 + 100 + 100 + 100)
print("valid_mutation_rate: ", valid_mutation_rate)

'''

cd /home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations
find . -type f -name "*.py" -exec sed -i 's/correct_persuasion_framework_final/correct_persuasion_framework_final/g' {} +
find . -type f -name "*.py" -exec sed -i 's/改/改/g' {} +

cd /home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/single_mutations
find . -type f -name "*.py" -exec sed -i 's/16_Foot-in-the-Door/16_Foot-in-the-Door/g' {} +
find . -type f -name "*.py" -exec sed -i 's/"Foot-in-the-Door"/"Foot-in-the-Door"/g' {} +

'''

# claude_3_haiku_mutation_avg_rouge1 = ( +  +  +  +  + ) / 6
# print("claude_3_haiku_mutation_avg_rouge1: ", claude_3_haiku_mutation_avg_rouge1, "\n")
# claude_3_haiku_mutation_avg_rougeL = ( +  +  +  +  + ) / 6
# print("claude_3_haiku_mutation_avg_rougeL: ", claude_3_haiku_mutation_avg_rougeL, "\n")
# gpt_4o_mini_mutation_avg_rouge1 = ( +  +  +  +  + ) / 6
# print("gpt_4o_mini_mutation_avg_rouge1: ", gpt_4o_mini_mutation_avg_rouge1, "\n")
# gpt_4o_mini_mutation_avg_rougeL = ( +  +  +  +  + ) / 6
# print("gpt_4o_mini_mutation_avg_rougeL: ", gpt_4o_mini_mutation_avg_rougeL, "\n")
# valid_mutation_rate = ( +  +  +  +  + ) / ( +  +  +  +  + )
# print("valid_mutation_rate: ", valid_mutation_rate)
