import functools
import importlib
import types
from collections import namedtuple
from importlib import resources
from typing import Dict, Optional, Type, TypeVar

T = TypeVar("T")

# Basic structure for storing information about one plugin
Plugin = namedtuple("Plugin", ("name", "func"))

# Dictionary with information about all registered plugins


class Registry():
    def __init__(self, intro) -> None:
        self._dict = {}
        self.intro = intro

    def register_from_dict(self,
                           name_to_func: Dict[str, T]):
        for k, v in name_to_func.items():
            self.register(v, name=k)

    def register(self,
                 func: T = None,
                 *,
                 name: Optional[str] = None
                 ) -> T:

        if name is None:
            name = func.__name__

        if self._dict.get(name) is not None:
            raise KeyError(f'Name {name} is already used, try another name!')

        self._dict[name] = func
        return func

    def __call__(self,
                 name: str,
                 *args,
                 **kwargs):
        ret = self._dict.get(name)
        if ret is None:
            _registry = {k.lower(): v for k, v in self._dict.items()}
            ret = _registry.get(name.lower())
            if ret is None:
                raise KeyError(f'Unknown {name} is called!')
            if len(args) > 0 or len(kwargs) > 0:
                # if args and kwargs is specified, instantiate
                ret = ret(*args, **kwargs)
        return ret
