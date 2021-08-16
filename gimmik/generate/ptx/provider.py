# -*- coding: utf-8 -*-

from gimmik.utils import new_line

class PTXProvider(object):

    def bra_tgt(self, tgt):
        return f'{tgt}:\n'

    @new_line
    def bar_sync(self, b, num=None):
        assert b < 16
        num_threads = f', {num}' if num is not None else ''
        return f'bar.sync {b}{num_threads}'

    @new_line
    def ret(self):
        return 'ret'
