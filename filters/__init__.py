filters = {}

def register_filter(name, filter):
    global filters
    filters[name] = filter

def get_filter(name):
    return filters.get(name, None)

from . import grayscale
from . import blur
from . import color
from . import roll
