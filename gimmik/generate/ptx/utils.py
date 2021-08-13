# -*- coding: utf-8 -*-

import functools

def typein(t):
    def typein_inner(func):
        @functools.wraps(func)
        def wrapper(self, d, *args, **kwargs):
            assert d.type in t
            return func(self, d, *args, **kwargs)
        
        return wrapper
    
    return typein_inner
