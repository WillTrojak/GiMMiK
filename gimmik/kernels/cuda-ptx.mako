# -*- coding: utf-8 -*-

<%namespace module='gimmik.generate.ptx.generate' name='p'/>

% if split is None:
    ${p.generator(sm, mat, beta, funcn, dtype, block_dim)}
% else:
    ${p.generator(sm, mat, beta, funcn, dtype, block_dim, split, rep, shr_max)}
% endif
