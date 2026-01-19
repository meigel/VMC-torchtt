# -*- coding: utf-8 -*-
from __future__ import division, print_function
import argparse, os, json, time

# def get_path():
#     parser = argparse.ArgumentParser(description='parse json')
#     parser.add_argument('path', help='path to the directory')
#     args = parser.parse_args()

#     path = args.path
#     if not os.path.isdir(path):
#         raise IOError("'{}' is not a valid directory".format(path))

#     return path

def get_paths(descr, *paths):
    parser = argparse.ArgumentParser(description=descr)
    for pt, pth in paths:
        parser.add_argument(pt, help=pth)
    args = parser.parse_args()

    out = []
    for pt, _ in paths:
        path = getattr(args, pt)
        if not os.path.isdir(path):
            raise IOError("'{}' is not a valid directory".format(path))
        out.append(path)

    return out

def load_info(path):
    fn = os.path.join(path, 'info.json')
    try:
        with open(fn, 'r') as f:
            info = json.load(f)
    except:
        raise IOError("'{}' is not a valid JSON file".format(fn))

    return info

def load_problem(info):
    problem_name = info['problem']['name']
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    print("{} Loading problem: '{}'".format(ts, problem_name))
    problem_module = __import__('problem.'+problem_name)
    return getattr(problem_module, problem_name)

def load_sampler(info):
    distribution = info['sampling']['distribution']
    if distribution not in ['uniform', 'normal']:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        print("{}, ERROR: '{}' is not a valid distribution".format(ts, distribution))
        exit(1)
    strategy = info['sampling']['strategy']
    if strategy not in ['random', 'sobol']:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        print("{} ERROR: '{}' is not a valid strategy".format(ts, strategy))
        exit(1)
    from sampler import RVSampler
    return RVSampler(info['expansion']['size'], distribution, strategy)

if __name__=="__main__":
    path = get_path()
    print(path)
    info = load_info(path)
    print(info)
    problem = load_problem(info)
    print(problem)
    sampler = load_sampler(info)
    print(sampler)
    FEM = problem.setup_FEM(info)
    print(FEM)
    ys = sampler(2)
    print(ys.shape)
    a0 = problem.evaluate_a(ys[0])
    print(a0.shape)
    u0 = problem.evaluate_u(a0)
    print(u0.shape)
