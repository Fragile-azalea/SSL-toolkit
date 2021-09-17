from DeSSL import Registry

SEMI_TRAINER_REGISTRY = Registry('semi_trainer')

while True:
    from .SemiBase import SemiBase
    from .Ladder import Ladder
    from .MeanTeacher import MeanTeacher
    # from .utils import unroll
    # from .InterpolationConsistency import InterpolationConsistency
    # from .AdversariallyLearnedInference import AdversariallyLearnedInference
    # from .MixMatch import MixMatch
    # from .VariationalAutoEncoder import VariationalAutoEncoder
    break
