filters = {}

def register_filter(name, filter):
    global filters
    filters[name] = filter

def get_filter(name):
    return filters.get(name, None)

def get_filters(filter_list):
    image_filters = []
    for filters_item in filter_list:
        if type(filters_item) == str:
            image_filters.append(get_filter(filters_item))
        if type(filters_item) == list:
            filter_name = filters_item[0]

            params = filters_item[1:]
            _args = []
            _kwargs = {}
            if len(params) == 1 and type(params[0]) == list:
                # ["filtername", ["value1", "value2"]]
                _args = params[0]
            elif len(params) == 1 and type(params[0]) == dict:
                # ["filtername", {param1: "value1", "param2": "value2"}]
                _kwargs = params[0]
            else:
                # ["filtername", "value1", "value2"]
                _args = params

            _image_filter = get_filter(filter_name)
            if not _image_filter:
                continue
            def filter_with_parameters(_image_filter=_image_filter,
                    _args=_args, _kwargs=_kwargs, *args, **kwargs):
                # Using default parameters is neccessary to work with
                # a copy of _image_filter, _args and _kwargs instead of
                # a reference
                args = list(args)
                for arg in _args:
                    args.append(arg)
                for key in _kwargs:
                    if not key in kwargs:
                        kwargs[key] = _kwargs[key]
                return _image_filter(*args, **kwargs)

            image_filters.append(filter_with_parameters)
    return image_filters

def apply_filters(frame, image_filters):
    for image_filter in image_filters:
        try:
            frame = image_filter(frame=frame)
        except TypeError:
            # caused by a wrong number of arguments in the config
            pass
    return frame

from . import grayscale
from . import blur
from . import gaussian_blur
from . import color
from . import roll
from . import transparency
from . import noise
from . import transformations
