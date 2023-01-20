## Requirements

Python 3.8

See `requirements.txt`.

## Model Loading

Models should be trained and saved before loading in a similar was as can be seen in `run_model.py`.

To load trained models:

```python
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
```

For command sequence representation you will also have to add some new tokens to the vocabulary.

```python
tokenizer.add_tokens(['</[DEL]/>', '</[INS]/>', '</[LOC]/>'])
```
