
# ---------------------------- .sh for run.py ---------------------------- #

python3 ./run.py \
        --prompt_save_folder ./prompt/ \
        --result_save_folder ./result/ \
        --recommend_num 10\
        --prompt_format 3 \
        --experiment 1 \
        --api_key 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


'''
# ---------------------------- .sh for post.py ---------------------------- #
python3 ./post.py \
        --recommend_num 10\
        --prompt 1 \
        --experiment 2 # total number of experiments for prompt
'''