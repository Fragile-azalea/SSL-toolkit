from homura import Registry

SEMI_TRAINER_REGISTRY = Registry('semi_trainer')

while True:
    from .utils import unroll
    from .Ladder import Ladder
    from .MeanTeacher import MeanTeacher
    from .InterpolationConsistency import InterpolationConsistency
    from .AdversariallyLearnedInference import AdversariallyLearnedInference
    from .MixMatch import MixMatch
    break
