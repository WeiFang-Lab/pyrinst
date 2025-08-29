from .modes import Data, Minimum, TransitionState, Instanton
from .opt.optimizers import optimizers

modes_registry: dict[str, type[Data]] = {'min': Minimum, 'ts': TransitionState, 'inst': Instanton}
