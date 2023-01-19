from dataclasses import dataclass

from aprkits.annotations import dictclass


@dataclass
@dictclass
class ValidationResult:
    loss: float = None
    accuracy: float = None
    floor_accuracy: float = None
    bleu: float = None
    code_bleu: float = None
