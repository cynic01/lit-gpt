from analyse_oasst1 import get_conversation_text
import sys
import torch
from torch.utils.data import random_split
from pathlib import Path
from tqdm import tqdm
from itertools import chain

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

IGNORE_INDEX = -1

def prepare(
    destination_path: Path = Path("data/oasst1"),
    checkpoint_dir: Path = Path("checkpoints/EleutherAI/pythia-1b-deduped"),
    eval_split_fraction: float = 0.1,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = True,  
    train_data_file: str = 'oasst1_train.pkl',
    test_data_file: str = 'oasst1_validation.pkl'
) -> None:
    """Prepare the Open Assistant dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = Tokenizer(checkpoint_dir)
    # tokenizer.processor.enable_truncation(max_seq_length, direction='left') #TODO: truncation probably should be at turn-level

    data = get_conversation_text(destination_path / train_data_file)
    train_set, eval_set = random_split(data,
                                       [1 - eval_split_fraction, eval_split_fraction],
                                       generator=torch.Generator().manual_seed(seed))
    test_set = get_conversation_text(destination_path / test_data_file)

    print("Processing train split ...")
    train_set = [_ for sample in tqdm(train_set) for _ in prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing eval split ...")
    eval_set = [_ for sample in tqdm(eval_set) for _ in prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)]
    torch.save(eval_set, destination_path / "eval.pt")

    print("Processing test split ...")
    test_set = [_ for sample in tqdm(test_set) for _ in prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) ]
    torch.save(test_set, destination_path / "test.pt")

    print(f"train has {len(train_set):,} samples")
    print(f"eval has {len(eval_set):,} samples")
    print(f"test has {len(test_set):,} samples")

def prepare_sample(example: list[tuple], tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True, num_turns: int = 1, symmetrical: bool = False):
    """Processes a single sample.

    Each sample is a multi-turn conversation, each turn is a tuple consisting of:
    - role: either prompter or assistant
    - message: could be self-contained or following up on a previous message
    
    Possible modes:
    - single turn exchanges: slice a conversation into a collection of independent prompt/assistant response pairs,  
    disregarding possible dependence on previous context, where response serves as labels. TODO
    - progressively longer context leading up to each assistant response, truncated to maximum context window.
    - fixed number of turns before the assistant's response. TODO
    - symmetrical: roles of the prompter assistant are allowed to be swapped TODO

    This function processes this data to produce a prompt text and a label for
    supervised training. 

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    for i in range(len(example)):
        speaker, message = example[i]
        # print(speaker, message[:40])
        if speaker == 'assistant': 
            if num_turns > 1:
                encoded_full_prompt, encoded_full_prompt_and_response = generate_prompt(example, i, tokenizer, max_length)
            else:
                encoded_full_prompt, encoded_full_prompt_and_response = generate_prompt(example[i-num_turns:i+1], num_turns, tokenizer, max_length)
            
            # The labels are the full prompt with response, but with the prompt masked out
            labels = encoded_full_prompt_and_response.clone()
            if mask_inputs:
                labels[: len(encoded_full_prompt)] = IGNORE_INDEX
            
            yield {
                "input_ids": encoded_full_prompt_and_response,
                "input_ids_no_response": encoded_full_prompt,
                "labels": labels,
            }
            
def generate_prompt(turn_sequence, current_turn, tokenizer, max_length):
    '''
    Input: all turns and current_turn idx where speaker is assistant (will become label).
    Should handle tokenization, truncation and ellipsis.
    First, tokenize current turn, if it doesn't fit - truncate at end (no ellipsis).
    At a minimum, include previous non-assistant turn, possibly with ellipsis.
    Then, gradually add previous turns starting with most recent, if that doesn't fully fit - ellipsis keeping speaker.
    '''
    current_speaker, current_message = turn_sequence[current_turn]
    if current_speaker == 'prompter':
        current_speaker = 'user'
    encoded_full_prompt_and_response = tokenizer.encode(f'<|{current_speaker}|>{current_message}', eos=True)
    assert not (len(encoded_full_prompt_and_response) > max_length and len(turn_sequence) == 1)
    response_length = len(encoded_full_prompt_and_response)

    at_least_one_full_turn_prompt = False
    for prev_speaker, prev_message in reversed(turn_sequence[:current_turn]):
        if prev_speaker == 'prompter':
            prev_speaker = 'user'
        prev_message_tokenized = tokenizer.encode(f'<|{prev_speaker}|>{prev_message}')
        
        if len(encoded_full_prompt_and_response) + len(prev_message_tokenized) > max_length:
            if not at_least_one_full_turn_prompt:
                encoded_full_prompt_and_response = torch.cat([
                    prev_message_tokenized[:max_length], 
                    encoded_full_prompt_and_response[:max(0, max_length - len(prev_message_tokenized))]
                ])
                response_length = len(encoded_full_prompt_and_response) - len(prev_message_tokenized)
                
                break
            else:
                ellipsis = tokenizer.encode('...\n')
                if max_length - len(encoded_full_prompt_and_response) - len(ellipsis) > 0:
                    encoded_full_prompt_and_response = torch.cat([
                        prev_message_tokenized[:max_length - len(encoded_full_prompt_and_response) - len(ellipsis)],
                        ellipsis,
                        encoded_full_prompt_and_response
                    ])         
                break       
        else:
            encoded_full_prompt_and_response = torch.cat([prev_message_tokenized, encoded_full_prompt_and_response])
        at_least_one_full_turn_prompt = True
    
    try:
        assert encoded_full_prompt_and_response.size()[0] <= max_length
    except AssertionError as e:
        print (e, encoded_full_prompt_and_response)
        raise e
        
    return encoded_full_prompt_and_response[:max(0, len(encoded_full_prompt_and_response) - response_length + len(tokenizer.encode(f'<|assistant|>')))].clone(), encoded_full_prompt_and_response
        
        
if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
