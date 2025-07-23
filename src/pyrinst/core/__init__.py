from .modes import Data, Minimum, TransitionState

modes_registry: dict[str, type[Data]] = {'min': Minimum, 'ts': TransitionState}
