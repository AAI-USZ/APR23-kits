## Requirements

Python 3.8

## Model Loading

Load trained models:

```python
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained('Salesforce/codet5-base')
```

For command sequence representation you will also have to add some new tokens to the vocabulary.

```python
tokenizer.add_tokens(['</[DEL]/>', '</[INS]/>', '</[LOC]/>'])
```
