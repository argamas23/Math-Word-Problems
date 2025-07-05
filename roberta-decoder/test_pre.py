from transformers import RobertaTokenizerFast as RobertaTokenizer
from src.pre_data import transfer_num # Adjust path if necessary

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

sample_data = [
    {
        "segmented_text": "Conner has 25000 dollars in his bank account. Every month he spends 1500 dollars. He does not add money to the account. How much money will Conner have in his account after 8 months?",
        "equation": "x = 25000.0 - (1500.0 * 8.0)"
    },
    {
        "segmented_text": "A rectangle is 10 cm wide and 5 cm tall. What is its area?",
        "equation": "x = 10 * 5"
    }
]

pairs, generate_nums, copy_nums = transfer_num(sample_data, tokenizer)

for i, p in enumerate(pairs):
    print(f"--- Problem {i+1} ---")
    input_ids, out_seq, original_nums, roberta_num_pos = p
    print(f"Text: {sample_data[i]['segmented_text']}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"Input IDs: {input_ids}")
    print(f"Original Nums: {original_nums}")
    print(f"RoBERTa Num Pos: {roberta_num_pos}")
    # Verification:
    for num_idx, token_idx in enumerate(roberta_num_pos):
        if token_idx != -1 and token_idx < len(input_ids):
             # Get the token(s) starting at token_idx that correspond to original_nums[num_idx]
             # This part can be tricky as a number might span multiple tokens
             # A simple check is to see if the token at token_idx is part of the number
            print(f"  Num '{original_nums[num_idx]}' (at text pos) starts at token index {token_idx}: '{tokenizer.convert_ids_to_tokens([input_ids[token_idx]])[0]}'")
        else:
            print(f"  Num '{original_nums[num_idx]}' could not be mapped to a token position.")
    print(f"Equation: {sample_data[i]['equation']}")
    print(f"Output Seq: {out_seq}")
    print("\n")

print(f"Generate Nums: {generate_nums}")
print(f"Copy Nums: {copy_nums}")