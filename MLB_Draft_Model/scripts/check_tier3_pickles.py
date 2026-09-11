#!/usr/bin/env python3
"""Quick check of Tier 3 pickle files"""
import pickle, json

BASE = '/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model'

for pt in ['hitter', 'pitcher']:
    fpath = BASE + '/models/artifacts_full/tier3_mlb_' + pt + '.pkl'
    with open(fpath, 'rb') as f:
        d = pickle.load(f)
    print(f'=== TIER 3 {pt.upper()} ===')
    print('  Keys:', list(d.keys()))
    print('  Features:', d['features'])
    print('  Num features:', len(d['features']))
    print('  Model type:', type(d['model']))
    print('  Coef shape:', d['model'].coef_.shape)
    print('  Coef:', d['model'].coef_[0])
    print('  Intercept:', d['model'].intercept_[0])
    rates = d['round_rates']
    print('  Round rates keys:', sorted(rates.keys())[:10])
    print()
