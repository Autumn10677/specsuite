from importlib.metadata import version, PackageNotFoundError
from .cosmic_rays import *  # noqa
from .throughput import *  # noqa
from .extraction import *  # noqa
from .warping import *  # noqa
from .loading import *  # noqa
from .wavecal import *  # noqa
from .widget import *  # noqa
from .utils import *  # noqa

__all__ = []
try:
    __version__ = version("specsuite")
except PackageNotFoundError:
    __version__ = "0.0.0"
