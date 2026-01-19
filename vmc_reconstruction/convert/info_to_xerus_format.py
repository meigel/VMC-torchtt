# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import argparse, os, sys, json
from load_info import load_info, load_problem


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to the directory containing the `info.json`')
    parser.add_argument('STORE_PATH', help='path to the directory to which the `info.json` will be saved')
    args = parser.parse_args()

    print("Reading Info from '{}' ".format(os.path.join(args.PATH, 'info.json')))
    info = load_info(args.PATH)
    problem = load_problem(info)
    fem_space = problem.setup_space(info)['V']
    fem_dofs = len(fem_space.dofmap().dofs())
    dist = { 'uniform': 0, 'normal': 1 }[ info['sampling']['distribution'] ]

    ret = {
        'problem type': info['problem']['name'],
        'fem dofs': fem_dofs,
        'expansion size': info['expansion']['size'],
        'distribution': dist,
        'mesh size': info['fe']['mesh size']
    }

    if not os.path.exists(args.STORE_PATH):
        os.makedirs(args.STORE_PATH)
    out_file = os.path.join(args.STORE_PATH, 'info.json')
    print("Writing Output to '{}' ".format(out_file))
    with open(out_file, 'w') as f:
        json.dump(ret, f)
