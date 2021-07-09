import random

import torch

from typing import List
import itertools

from vocab import Vocab, transfer_vocab_index
from asts.ast_parser import get_single_ast
import vars


def collate_fn(batch, task, code_vocab, nl_vocab, ast_vocab,
               max_code_len, max_ast_len, max_nl_len):
    """
    Data collator function.

    Args:
        batch (list):
        task (str):
        code_vocab (Vocab):
        nl_vocab (Vocab):
        ast_vocab (Vocab):
        max_code_len (int):
        max_ast_len (int):
        max_nl_len (int):

    Returns:
        dict: model inputs
    """
    model_inputs = {}

    if task == vars.TASK_CODE_AST_PREDICTION:
        code_raw = [sample[0] for sample in batch]
        ast_raw = [sample[1] for sample in batch]
        nl_raw = [sample[2] for sample in batch]
        is_ast = [sample[3] for sample in batch]

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(code_raw=code_raw,
                                                                                            code_vocab=code_vocab,
                                                                                            max_code_len=max_code_len,
                                                                                            ast_raw=ast_raw,
                                                                                            ast_vocab=ast_vocab,
                                                                                            max_ast_len=max_ast_len,
                                                                                            nl_raw=nl_raw,
                                                                                            nl_vocab=nl_vocab,
                                                                                            max_nl_len=max_nl_len)
        model_inputs['labels'] = torch.tensor([[a] for a in is_ast], dtype=torch.long)
        model_inputs['is_cls'] = True

    elif task == vars.TASK_NEXT_CODE_PREDICTION:
        lang_raw = [sample[0] for sample in batch]
        code_raw = [sample[1] for sample in batch]
        nl_raw = [sample[2] for sample in batch]

        former_codes = []
        latter_codes = []
        former_asts = []
        for lang, code in zip(lang_raw, code_raw):

            pos = random.randint(1, len(code) - 2)
            former_code = code[: pos]
            latter_code = code[pos:]

            former_ast = get_single_ast(lang=lang, source=former_code)

            former_codes.append(former_code)
            latter_codes.append(latter_code)
            former_asts.append(former_ast)

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(code_raw=former_codes,
                                                                                            code_vocab=code_vocab,
                                                                                            max_code_len=max_code_len,
                                                                                            ast_raw=former_asts,
                                                                                            ast_vocab=ast_vocab,
                                                                                            max_ast_len=max_ast_len,
                                                                                            nl_raw=nl_raw,
                                                                                            nl_vocab=nl_vocab,
                                                                                            max_nl_len=max_nl_len)
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=latter_codes, vocab=code_vocab, processor=Vocab.sos_processor, max_len=max_code_len)
        model_inputs['labels'], _ = get_batch_inputs(batch=latter_codes,
                                                     vocab=code_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=max_code_len)
        model_inputs['is_gen'] = True

    elif task == vars.TASK_METHOD_NAME_PREDICTION:
        code_raw = [sample[0] for sample in batch]
        ast_raw = [sample[1] for sample in batch]
        nl_raw = [sample[2] for sample in batch]

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(code_raw=code_raw,
                                                                                            code_vocab=code_vocab,
                                                                                            max_code_len=max_code_len,
                                                                                            ast_raw=ast_raw,
                                                                                            ast_vocab=ast_vocab,
                                                                                            max_ast_len=max_ast_len)
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=nl_raw, vocab=nl_vocab, processor=Vocab.sos_processor, max_len=max_nl_len)
        model_inputs['labels'], _ = get_batch_inputs(batch=nl_raw,
                                                     vocab=nl_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=max_nl_len)
        model_inputs['is_gen'] = True

    elif task == vars.TASK_SUMMARIZATION:
        code_raw = [sample[0] for sample in batch]
        ast_raw = [sample[1] for sample in batch]
        name_raw = [sample[2] for sample in batch]
        nl_raw = [sample[3] for sample in batch]

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(code_raw=code_raw,
                                                                                            code_vocab=code_vocab,
                                                                                            max_code_len=max_code_len,
                                                                                            ast_raw=ast_raw,
                                                                                            ast_vocab=ast_vocab,
                                                                                            max_ast_len=max_ast_len,
                                                                                            nl_raw=name_raw,
                                                                                            nl_vocab=nl_vocab,
                                                                                            max_nl_len=max_nl_len)
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=nl_raw, vocab=nl_vocab, processor=Vocab.sos_processor, max_len=max_nl_len)
        model_inputs['labels'], _ = get_batch_inputs(batch=nl_raw,
                                                     vocab=nl_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=max_nl_len)
        model_inputs['is_gen'] = True

    elif task == vars.TASK_TRANSLATION:
        code_raw = [sample[0] for sample in batch]
        ast_raw = [sample[1] for sample in batch]
        name_raw = [sample[2] for sample in batch]
        target_raw = [sample[3] for sample in batch]

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(code_raw=code_raw,
                                                                                            code_vocab=code_vocab,
                                                                                            max_code_len=max_code_len,
                                                                                            ast_raw=ast_raw,
                                                                                            ast_vocab=ast_vocab,
                                                                                            max_ast_len=max_ast_len,
                                                                                            nl_raw=name_raw,
                                                                                            nl_vocab=nl_vocab,
                                                                                            max_nl_len=max_nl_len)
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=target_raw, vocab=code_vocab, processor=Vocab.sos_processor, max_len=max_code_len)
        model_inputs['labels'], _ = get_batch_inputs(batch=target_raw,
                                                     vocab=code_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=max_code_len)
        model_inputs['is_gen'] = True

    elif task == vars.TASK_SEARCH:
        pass

    return model_inputs


def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None):
    """
    Encode the given batch to input to the model.

    Args:
        batch (list[str]): Batch of sequence,
            each sequence is represented by a string or list of tokens
        vocab (Vocab): Vocab of the batch
        processor (tokenizers.processors.PostProcessor): Optional, post-processor method
        max_len (int): Optional, the maximum length of each sequence

    Returns:
        (torch.LongTensor): Tensor of batch, [B, T]

    """
    # set post processor
    vocab.tokenizer.post_processor = processor
    # set truncation
    if max_len:
        vocab.tokenizer.enable_truncation(max_length=max_len)
    else:
        vocab.tokenizer.no_truncation()
    # encode batch
    encoded_batch = vocab.encode_batch(batch, pad=True, max_length=max_len)
    # to tensor
    inputs = torch.tensor([encoded.ids for encoded in encoded_batch], dtype=torch.long)
    padding_mask = torch.tensor([encoded.attention_mask for encoded in encoded_batch], dtype=torch.long)

    return inputs, padding_mask


def get_concat_batch_inputs(code_raw, code_vocab, max_code_len, ast_raw, ast_vocab, max_ast_len,
                            nl_raw=None, nl_vocab=None, max_nl_len=None):
    code_vocab.tokenizer.post_processor = Vocab.sep_processor
    code_vocab.tokenizer.enable_truncation(max_length=max_code_len)
    code_encoded = code_vocab.encode_batch(code_raw, pad=False)

    ast_vocab.tokenizer.post_processor = Vocab.sep_processor if nl_raw else Vocab.eos_processor
    ast_vocab.tokenizer.enable_truncation(max_length=max_ast_len)
    ast_encoded = ast_vocab.encode_batch(ast_raw, pad=False)

    if nl_raw:
        nl_vocab.tokenizer.post_processor = Vocab.eos_processor
        nl_vocab.tokenizer.enable_truncation(max_length=max_nl_len)
        nl_encoded = nl_vocab.encode_batch(batch=nl_raw, pad=False)

    concat_inputs = []
    concat_masks = []
    for index, (code, ast) in enumerate(zip(code_encoded, ast_encoded)):
        ids = code.ids + transfer_vocab_index(ast.ids, vocab=ast_vocab, base=len(code_vocab))
        masks = code.attention_mask + ast.attention_mask
        if nl_raw:
            ids += transfer_vocab_index(nl_encoded[index].ids, vocab=nl_vocab, base=len(code_vocab) + len(ast_vocab))
            masks += nl_encoded[index].attention_mask
        concat_inputs.append(ids)
        concat_masks.append(masks)

    concat_inputs = pad_batch(concat_inputs)
    concat_masks = pad_batch(concat_masks)

    return concat_inputs, concat_masks


def pad_batch(batch, pad_value=0):
    """
    Pad a list of sequence to a padded 2d tensor.

    Args:
        batch (list[list[int]]): List of sequence
        pad_value (int): Optional, fill value, default to 0.

    Returns:
        torch.Tensor: Padded tensor. [B, T].

    """
    batch = list(zip(*itertools.zip_longest(*batch, fillvalue=pad_value)))
    return torch.tensor([list(b) for b in batch]).long()