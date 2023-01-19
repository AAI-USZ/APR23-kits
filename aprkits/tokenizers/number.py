import json
from pathlib import Path
from typing import Optional, List, Dict

from transformers import PreTrainedTokenizer


class NumDict(dict):
    def __init__(self, base: int = 1):
        super().__init__()
        self.base = base

    def __getitem__(self, item):
        if isinstance(item, str):
            return int(item) + self.base
        elif isinstance(item, int):
            return str(item - self.base)
        raise NotImplementedError('Number dictionary is not implemented for anything other than (str, int).')


class NumberTokenizer(PreTrainedTokenizer):
    def __init__(
            self,
            pad_token_id: int = 0,
            unk_token_id: int = 0,
            cls_token_id: int = 0,
            sep_token_id: int = 0,
            bos_token_id: int = 0,
            eos_token_id: int = 0,
            model_max_length: int = None,
            **kwargs
    ):
        super().__init__(model_max_length=model_max_length, **kwargs)
        if 'base' in kwargs:
            self.tokenizer = NumDict(base=kwargs['base'])
        else:
            self.tokenizer = NumDict(base=pad_token_id + 1)
        self.model_input_names = ['input_ids']
        self.add_special_tokens({
            'unk_token': str(unk_token_id),
            'pad_token': str(pad_token_id),
            'cls_token': str(cls_token_id),
            'sep_token': str(sep_token_id),
            'bos_token': str(bos_token_id),
            'eos_token': str(eos_token_id)
        })
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def vocab_size(self) -> int:
        return 0

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.tokenizer[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer[index]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        print(f'{NumberTokenizer.__name__}\'s vocabulary cannot be saved since it has no real dictionary.\n'
              'An empty dictionary {} will be saved instead.')
        fname = Path(
            save_directory,
            f'{(filename_prefix + ".") if filename_prefix is not None else ""}vocab.json'
        )
        savedir = Path(save_directory)
        if not savedir.exists():
            savedir.mkdir(parents=True)
        with open(fname, 'w', encoding='utf-8') as fp:
            json.dump({}, fp, ensure_ascii=True, indent=2)
        return str(fname),
