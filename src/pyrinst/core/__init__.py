from .modes import Data, Minimum, TransitionState, Instanton

modes_registry: dict[str, type[Data]] = {'min': Minimum, 'ts': TransitionState, 'inst': Instanton}
