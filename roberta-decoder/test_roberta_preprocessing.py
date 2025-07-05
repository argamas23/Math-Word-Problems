#!/usr/bin/env python3
# Test script for RoBERTa preprocessing

import sys
sys.path.append('/home2/abhyudit.singh/working/rpkhs')

from transformers import RobertaTokenizerFast as RobertaTokenizer
from src.pre_data import load_raw_data, transfer_num

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print("Loaded RoBERTa tokenizer")

# Load a small sample of data
try:
    data = load_raw_data("/home2/abhyudit.singh/working/rpkhs/data/test.json")
    print(f"Loaded {len(data)} problems")
    
    # Take first 3 problems for testing
    sample_data = data[:3]
    
    # Test transfer_num function
    pairs, generate_nums, copy_nums = transfer_num(sample_data, tokenizer)
    print(f"Processed {len(pairs)} pairs")
    print(f"Generate nums: {generate_nums}")
    print(f"Copy nums: {copy_nums}")
    
    # Inspect first pair
    if pairs:
        input_ids, out_seq, original_nums, roberta_num_pos = pairs[0]
        print(f"\nFirst problem:")
        print(f"Input IDs: {input_ids}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
        print(f"Original nums: {original_nums}")
        print(f"RoBERTa num pos: {roberta_num_pos}")
        print(f"Output seq: {out_seq}")
        
        # Verify number mapping
        for i, pos in enumerate(roberta_num_pos):
            if pos != -1 and pos < len(input_ids):
                token = tokenizer.convert_ids_to_tokens([input_ids[pos]])[0]
                print(f"  Number '{original_nums[i]}' -> token position {pos}: '{token}'")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
