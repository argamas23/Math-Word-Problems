# coding: utf-8

import random
import json
import copy
import re

# Padding token ID
PAD_token = 0


class Lang:
    """
    Language class to manage vocabulary, word indices, and counts
    """
    def __init__(self):
        self.word2index = {}  # Map from words to indices
        self.word2count = {}  # Map from words to occurrence counts
        self.index2word = []  # Map from indices to words
        self.n_words = 0      # Total number of unique words
        self.num_start = 0    # Starting index for numbers

    def add_sen_to_vocab(self, sentence):
        """Add all words in a sentence to vocabulary"""
        for word in sentence:
            if re.search(r"N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):
        """Remove words below a minimum count threshold"""
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):
        """Build the input language vocabulary and dictionaries"""
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        
        self.word2index = {}
        self.n_words = len(self.index2word)
        
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_nums, copy_nums):
        """Build the output language vocabulary and dictionaries"""
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_nums + \
                        ["N" + str(i) for i in range(copy_nums)] + ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_nums, copy_nums):
        """Build the output language vocabulary specifically for tree generation"""
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_nums + \
                        ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):
    """
    Load raw data from JSON file
    
    Args:
        filename: Path to the JSON data file
        
    Returns:
        List of data dictionaries
    """
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # Every 7 lines is a complete JSON entry
            data_d = json.loads(js)
            # Clean up equations if needed
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
    return data


def remove_brackets(x):
    """
    Remove superfluous brackets from an expression
    
    Args:
        x: Expression string to process
        
    Returns:
        Processed expression string
    """
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def transfer_num(data):
    """
    Transfer numbers in data to their tokenized form
    
    Args:
        data: List of data entries
        
    Returns:
        processed_pairs: List of processed data pairs
        generate_nums: List of generated numbers
        copy_nums: Maximum number of copied numbers
    """
    print("Transferring numbers...")
    pattern = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]  # Remove "x="

        # Process input sequence and extract numbers
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
                
        # Update maximum number of copied numbers
        if copy_nums < len(nums):
            copy_nums = len(nums)

        # Handle fraction numbers specifically
        nums_fraction = []
        for num in nums:
            if re.search(r"\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):
            """
            Segment and tag numbers in the equation
            """
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
                    
            pos_st = re.search(r"\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
                
            for ss in st:
                res.append(ss)
            return res

        # Process equation with number tagging
        out_seq = seg_and_tag(equations)
        
        # Identify numbers that need to be generated
        for s in out_seq:
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        # Record positions of numbers in input sequence
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        
        # Create the final processed pair
        pairs.append((input_seq, out_seq, nums, num_pos))

    # Filter generated numbers to keep only frequent ones
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
            
    return pairs, temp_g, copy_nums


def indexes_from_sentence(lang, sentence, tree=False):
    """
    Convert a sentence to a list of indices
    
    Args:
        lang: Language object with word2index mapping
        sentence: List of tokens/words
        tree: Whether we're processing tree data
        
    Returns:
        List of indices representing the sentence
    """
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
            
    # Add EOS token if needed
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
        
    return res


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    """
    Prepare data for training and testing
    
    Args:
        pairs_trained: Training data pairs
        pairs_tested: Testing data pairs
        trim_min_count: Minimum word frequency threshold
        generate_nums: List of numbers to generate
        copy_nums: Maximum number of copied numbers
        tree: Whether to use tree-based processing
        
    Returns:
        input_lang: Input language object
        output_lang: Output language object
        train_pairs: Processed training pairs
        test_pairs: Processed testing pairs
    """
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:  # Check the last element for tree eligibility
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
            
    # Build vocabularies
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    # Process training pairs
    for pair in pairs_trained:
        num_stack = []
        # Create number stack tracking positions of numbers
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        # Reverse stack for processing
        num_stack.reverse()
        
        # Convert tokens to indices
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        
        # Add processed pair to training data
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                          pair[2], pair[3], num_stack))
                        
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    
    # Process testing pairs
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        
        # Convert tokens to indices
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        
        # Add processed pair to testing data
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                         pair[2], pair[3], num_stack))
                        
    print('Number of testing data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def pad_seq(seq, seq_len, max_length):
    """
    Pad a sequence to the specified maximum length
    
    Args:
        seq: Sequence to pad
        seq_len: Current length of sequence
        max_length: Target length after padding
        
    Returns:
        Padded sequence
    """
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


def prepare_train_batch(pairs_to_batch, batch_size):
    """
    Prepare batches for training
    
    Args:
        pairs_to_batch: Data pairs to batch
        batch_size: Size of each batch
        
    Returns:
        Batched data in format suitable for training
    """
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # Shuffle the pairs
    
    # Initialize batch storage
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []
    num_pos_batches = []
    num_size_batches = []
    
    # Create batches
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    # Process each batch
    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        
        # Extract sequence lengths
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        
        # Get maximum lengths for padding
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        
        # Initialize batch data
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        
        # Process each example in the batch
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            
        # Add processed data to batch lists
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches