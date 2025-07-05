# coding: utf-8
import time
import torch # type: ignore
import argparse
import pathlib
import os
import json
import random
import numpy as np
from src.pre_data import *
from src.train_and_evaluate import *
from models.basic_models import *
from models.tree_models import *

# Set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Parse command line arguments
parser = argparse.ArgumentParser(description='PyTorch Seq2Tree')
parser.add_argument('--model_path', type=str, default='model/checkpoint',
                   help='path to save the model')
parser.add_argument('--resume', type=str, default='', 
                   help='resume from previously saved checkpoint')
parser.add_argument('--embedding_size', type=int, default=128,
                   help='embedding size for encoder and decoder')
parser.add_argument('--hidden_size', type=int, default=512,
                   help='hidden size for encoder and decoder')
parser.add_argument('--depth', type=int, default=2,
                   help='number of layers in encoder and decoder')
parser.add_argument('--beam_size', type=int, default=5,
                   help='beam size for beam search')
parser.add_argument('--dropout', type=float, default=0.5,
                   help='dropout rate')
parser.add_argument('--batch_size', type=int, default=8,
                   help='batch size')
parser.add_argument('--n_epochs', type=int, default=10,
                   help='number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                   help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                   help='weight decay')
parser.add_argument('--clip', type=float, default=5.0,
                   help='gradient clipping')
parser.add_argument('--warmup', type=int, default=0,
                   help='number of iterations for warmup')
parser.add_argument('--data_path', type=str, default='data/Math_23K.json',
                   help='path to dataset')
parser.add_argument('--train_test_ratio', type=float, default=0.8,
                   help='ratio of train to test data')
parser.add_argument('--word_min_count', type=int, default=5,
                   help='minimum count of word occurrence')
parser.add_argument('--evaluate', action='store_true', 
                   help='set this flag to evaluate the model')
parser.add_argument('--test_path', type=str, default='',
                   help='path to test dataset')

args = parser.parse_args()

def prepare_dataset():
    """
    Prepare train and test datasets
    """
    train_dataset_filename = args.data_path
    test_dataset_filename = args.test_path
    
    # Check if data paths exist
    data_dir = os.path.dirname(train_dataset_filename)
    model_dir = os.path.dirname(args.model_path)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data from {}".format(train_dataset_filename))
    data = load_raw_data(train_dataset_filename)
    
    # Split data if no test path is provided
    if not test_dataset_filename:
        train_size = int(len(data) * args.train_test_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
    else:
        train_data = data
        test_data = load_raw_data(test_dataset_filename)
    
    print("Train size: {}, Test size: {}".format(len(train_data), len(test_data)))
    
    # Process data: transform numbers to tokens
    pairs_trained, generate_nums, copy_nums = transfer_num(train_data)
    pairs_tested, _, _ = transfer_num(test_data)
    
    # Create input and output lang objects
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(
        pairs_trained, pairs_tested, args.word_min_count, generate_nums, copy_nums, tree=True
    )
    
    print("Unique tokens in input: {}".format(input_lang.n_words))
    print("Unique tokens in output: {}".format(output_lang.n_words))
    print("Max. equation length: {}".format(max(len(pair[2]) for pair in train_pairs)))
    
    # Generate number list
    num_list = list(generate_nums)
    num_list.sort()
    
    return input_lang, output_lang, train_pairs, test_pairs, generate_nums, copy_nums, num_list

def build_tree_models(input_lang, output_lang, generate_nums, copy_nums):
    """
    Build tree-based models: encoder, decoder, generator, etc.
    """
    # Create encoder
    encoder = EncoderSeq(
        input_size=input_lang.n_words,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_layers=args.depth,
        dropout=args.dropout
    )
    
    # Create prediction module
    predict = Prediction(
        hidden_size=args.hidden_size,
        op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
        input_size=len(generate_nums),
        dropout=args.dropout
    )
    
    # Create generation module
    generate = GenerateNode(
        hidden_size=args.hidden_size,
        op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
        embedding_size=args.embedding_size,
        dropout=args.dropout
    )
    
    # Create merging module
    merge = Merge(
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        dropout=args.dropout
    )
    
    # Use CUDA if available
    if USE_CUDA:
        encoder = encoder.cuda()
        predict = predict.cuda()
        generate = generate.cuda()
        merge = merge.cuda()
    
    return encoder, predict, generate, merge

def train_tree_model(input_lang, output_lang, train_pairs, test_pairs, generate_nums, copy_nums, num_list):
    """
    Train the tree-based model
    """
    # Build model components
    encoder, predict, generate, merge = build_tree_models(input_lang, output_lang, generate_nums, copy_nums)
    
    # Create optimizers
    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    predict_optimizer = torch.optim.Adam(
        predict.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generate_optimizer = torch.optim.Adam(
        generate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    merge_optimizer = torch.optim.Adam(
        merge.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Load checkpoint if resume flag is set
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder.load_state_dict(checkpoint['encoder'])
            predict.load_state_dict(checkpoint['predict'])
            generate.load_state_dict(checkpoint['generate'])
            merge.load_state_dict(checkpoint['merge'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    
    # Train for n_epochs
    for epoch in range(start_epoch, args.n_epochs):
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
        num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, args.batch_size)
        
        print("Epoch %d, Train data size: %d" % (epoch + 1, len(input_lengths)))
        start = time.time()
        
        # Train on batches
        for idx in range(len(input_lengths)):
            # Get batch data
            input_batch = input_batches[idx]
            input_length = input_lengths[idx]
            output_batch = output_batches[idx]
            output_length = output_lengths[idx]
            num_batch = nums_batches[idx]
            num_stack_batch = num_stack_batches[idx]
            num_pos_batch = num_pos_batches[idx]
            num_size_batch = num_size_batches[idx]
            
            # Train on this batch
            loss = train_tree(
                input_batch, input_length, output_batch, output_length, 
                num_stack_batch, num_size_batch, generate_nums,
                encoder, predict, generate, merge,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                output_lang, num_pos_batch
            )
            
            loss_total += loss
            
            if (idx+1) % 500 == 0:
                current = time.time()
                elapsed = current - start
                print("Epoch %d, Batch %d/%d, Loss: %.4f, Time: %ds" % 
                      (epoch + 1, idx + 1, len(input_lengths), loss, elapsed))
                start = current
        
        # Print epoch summary
        print("Epoch %d, Loss: %.4f" % (epoch + 1, loss_total / len(input_lengths)))
        
        # Evaluate on test set
        print("Evaluating on test set...")
        value_acc = 0
        equation_acc = 0
        eval_total = 0
        start = time.time()
        
        # Sample a subset of test data for evaluation during training
        test_subset = random.sample(test_pairs, min(500, len(test_pairs)))
        for test_pair in test_subset:
            input_batch = test_pair[0]
            input_length = test_pair[1]
            output_batch = test_pair[2]
            output_length = test_pair[3]
            num_batch = test_pair[4]
            num_pos_batch = test_pair[5]
            num_stack_batch = test_pair[6]
            
            # Run beam search
            test_res = evaluate_tree(
                input_batch, input_length, generate_nums, 
                encoder, predict, generate, merge,
                output_lang, num_pos_batch, beam_size=args.beam_size
            )
            
            # Check if result matches
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                test_res, output_batch, output_lang, num_batch, num_stack_batch
            )
            
            if val_ac:
                value_acc += 1
            if equ_ac:
                equation_acc += 1
            eval_total += 1
        
        # Calculate accuracy metrics
        val_acc = float(value_acc) / eval_total
        equ_acc = float(equation_acc) / eval_total
        print("Test data size: %d, Value accuracy: %.4f, Equation accuracy: %.4f, Time: %ds" %
              (eval_total, val_acc, equ_acc, time.time() - start))
        
        # Save checkpoint if value accuracy is better
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'encoder': encoder.state_dict(),
                'predict': predict.state_dict(),
                'generate': generate.state_dict(),
                'merge': merge.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, args.model_path + '.ckpt')
            print("Checkpoint saved to {}".format(args.model_path + '.ckpt'))

def evaluate_model(input_lang, output_lang, test_pairs, generate_nums, copy_nums, num_list):
    """
    Evaluate the tree-based model on test set
    """
    # Build model components
    encoder, predict, generate, merge = build_tree_models(input_lang, output_lang, generate_nums, copy_nums)
    
    # Load model checkpoint
    if not os.path.isfile(args.model_path + '.ckpt'):
        print("No checkpoint found at '{}'".format(args.model_path + '.ckpt'))
        return
        
    checkpoint = torch.load(args.model_path + '.ckpt')
    encoder.load_state_dict(checkpoint['encoder'])
    predict.load_state_dict(checkpoint['predict'])
    generate.load_state_dict(checkpoint['generate'])
    merge.load_state_dict(checkpoint['merge'])
    print("Loaded checkpoint from '{}'".format(args.model_path + '.ckpt'))
    
    # Set models to evaluation mode
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    
    # Evaluate on all test data
    value_acc = 0
    equation_acc = 0
    eval_total = 0
    start = time.time()
    
    for test_pair in test_pairs:
        input_batch = test_pair[0]
        input_length = test_pair[1]
        output_batch = test_pair[2]
        output_length = test_pair[3]
        num_batch = test_pair[4]
        num_pos_batch = test_pair[5]
        num_stack_batch = test_pair[6]
        
        # Run beam search
        test_res = evaluate_tree(
            input_batch, input_length, generate_nums, 
            encoder, predict, generate, merge,
            output_lang, num_pos_batch, beam_size=args.beam_size
        )
        
        # Check if result matches
        val_ac, equ_ac, res, tar = compute_prefix_tree_result(
            test_res, output_batch, output_lang, num_batch, num_stack_batch
        )
        
        if val_ac:
            value_acc += 1
        if equ_ac:
            equation_acc += 1
        eval_total += 1
        
        # Print example results
        if eval_total % 100 == 0:
            print("Example %d: Input: %s" % (eval_total, 
                ' '.join([input_lang.index2word[i] for i in input_batch if i != 0])))
            print("Target: %s" % ' '.join([str(x) for x in tar]))
            print("Result: %s" % ' '.join([str(x) for x in res]))
            print("Value match: %s, Equation match: %s" % (val_ac, equ_ac))
            print()
    
    # Calculate overall accuracy metrics
    val_acc = float(value_acc) / eval_total
    equ_acc = float(equation_acc) / eval_total
    print("Test data size: %d, Value accuracy: %.4f, Equation accuracy: %.4f, Time: %ds" %
          (eval_total, val_acc, equ_acc, time.time() - start))
    
    return val_acc, equ_acc

def main():
    # Prepare data
    input_lang, output_lang, train_pairs, test_pairs, generate_nums, copy_nums, num_list = prepare_dataset()
    
    # Evaluate or train
    if args.evaluate:
        evaluate_model(input_lang, output_lang, test_pairs, generate_nums, copy_nums, num_list)
    else:
        train_tree_model(input_lang, output_lang, train_pairs, test_pairs, generate_nums, copy_nums, num_list)

if __name__ == "__main__":
    main()