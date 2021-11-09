# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import warnings
import torch
from os import truncate
from torch import nn
from dataclasses import dataclass

from torch.nn.modules import module
from datasets import load_dataset
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class DualSumDataSet(nn.Module):
    def __init__(
        self,
        tokenizer,
        train_file,
        validation_file,
        test_file,
        prefix='',
        line_by_line=True,
        pad_to_max_length=True,
        max_src_length=512,
        max_tgt_length=100,
        max_src_amr_length=None,
        max_tgt_amr_length=None,
        ignore_pad_token_for_loss=True,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.max_src_amr_length = max_src_amr_length if max_src_amr_length is not None else max_src_amr_length
        self.max_tgt_amr_length = max_tgt_amr_length if max_tgt_amr_length is not None else max_tgt_length

    def setup(self, stage='fit'):
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        datasets = load_dataset("mtpe.py", data_files=data_files)
        print('datasets:', datasets)
        column_names = datasets["train"].column_names
        print('colums:', column_names)
        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                src = examples['src']
                tgt = examples['tgt']
                pe = examples['pe']
                src_inputs = self.tokenizer(tgt, max_length=self.max_src_length, padding=padding, truncation=True)
                # Setup the tokenizer for targets
                labels = self.tokenizer(pe, max_length=self.max_tgt_length, padding=padding, truncation=True)
                
                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if padding == "max_length" and self.ignore_pad_token_for_loss:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                    ]
                    
                src_inputs["labels"] = labels["input_ids"]
                # model_inputs["tgt_ids"] = tgt["input_ids"]
                return src_inputs

            self.train_dataset = datasets["train"].map(
                tokenize_function,
                batched=True,
                remove_columns=["src", 'tgt', 'pe'],
            )
            self.valid_dataset = datasets["validation"].map(
                tokenize_function,
                batched=True,
                remove_columns=["src", 'tgt', 'pe'],
            )
            self.test_dataset = datasets["test"].map(
                tokenize_function,
                batched=True,
                remove_columns=["src", 'tgt', 'pe'],
            )


def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        
        # padding_func(features, padding_side=self.tokenizer.padding_side, pad_token_id==self.tokenizer.pad_token_id, key="tgt_ids")
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }


@dataclass
class DataCollatorForDual2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
            "amr_ids": features["amr_ids"],
            "amr_mask": features["amr_mask"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }


@dataclass
class DataCollatorForDual2AMR:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["amr_labels"] for feature in features] if "amr_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["amr_labels"]))
                feature["amr_labels"] = (
                    feature["amr_labels"] + remainder if padding_side == "right" else remainder + feature["amr_labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["amr_labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
            "amr_ids": features["amr_ids"],
            "amr_mask": features["amr_mask"],
            "labels": features["amr_labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }


# @dataclass
# class DataCollatorForDual2Dual:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.

#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         model (:class:`~transformers.PreTrainedModel`):
#             The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
#             prepare the `decoder_input_ids`

#             This is useful when using `label_smoothing` to avoid calculating loss twice.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:

#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence is provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.

#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
#     """

#     tokenizer: PreTrainedTokenizerBase
#     model: Optional[PreTrainedModel] = None
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100

#     def __call__(self, batch):
#         input_ids = torch.stack([x["input_ids"] for x in batch])
#         masks = torch.stack([x["attention_mask"] for x in batch])
#         amr_ids = torch.stack([x["amr_ids"] for x in batch])
#         amr_masks = torch.stack([x["amr_mask"] for x in batch])
#         target_ids = torch.stack([x["labels"] for x in batch])
#         target_amr_ids = torch.stack([x["amr_labels"] for x in batch])
#         y = trim_batch(target_ids, self.label_pad_token_id)
#         y_amr = trim_batch(target_amr_ids, self.label_pad_token_id)
#         source_ids, source_mask = trim_batch(
#             input_ids, self.tokenizer.pad_token_id, attention_mask=masks
#         )
#         src_amr_ids, src_amr_mask = trim_batch(
#             amr_ids, self.tokenizer.pad_token_id, attention_mask=amr_masks
#         )
#         if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
#             decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=y)
#         return {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "amr_ids": src_amr_ids,
#             "amr_mask": src_amr_mask,
#             "labels": y,
#             "decoder_input_ids": decoder_input_ids,
#         }
