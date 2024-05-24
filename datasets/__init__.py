from typing import Optional, Callable, Union

from .caltech256 import Caltech256
from .imagenette import ImageNette
from .nips2017 import NIPS2017


def datasets_factory(name: str, root: str, transform: Optional[Callable] = None) -> Union[
    Caltech256, NIPS2017, ImageNette]:
    if name == 'caltech256':
        return Caltech256(root, transform)
    elif name == 'nips2017':
        return NIPS2017(root, transform)
    elif name == 'imagenette':
        return ImageNette(root, transform)
    else:
        raise NotImplementedError(f"The parameter name={name} is not implemented.")
