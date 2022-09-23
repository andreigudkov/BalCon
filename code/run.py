#!/usr/bin/env python3

# Forbid numpy to use more than one thread
# !!! Place before import numpy !!!
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import pathlib
from datetime import datetime
from time import perf_counter

from environment import Environment, Settings
from balcon import BalCon
from firstfit import FirstFit
from algorithm import Solution, Score
from solver import FlowModel, AllocationModel, FlowModelRelaxed
from sercon import SerCon, SerConOriginal


def get_time():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def process_input_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True,
                        help='path to the .json with problem instance')
    parser.add_argument('--output', type=str, required=False,
                        help='path to store results')
    parser.add_argument('--algorithm', type=str, nargs='+', required=True,
                        help=f'algorithms to run, possible algorithms: '
                             f'{[alg.__name__ for alg in registry.values()]}')

    parser.add_argument('--wa', type=float, required=False, default=10,
                        help='weight for savings score')
    parser.add_argument('--wm', type=float, required=False, default=1,
                        help='weight for migration score')
    parser.add_argument('--f', type=float, required=False, default=4000,
                        help='max number of force steps for BalCon algorithm')
    parser.add_argument('--tl', type=float, required=True,
                        help='time limit for each algorithm')
    return parser.parse_args()


def setup_enviroment(args: argparse.Namespace, output_dir: str) -> Environment:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(output_dir + '/log.txt'),
            logging.StreamHandler()
        ]
    )


    env = Environment.load(args.problem)
    assert env.validate_mapping(env.mapping), "initial solution must be feasible"
    env.save(output_dir + '/problem.json')

    return env 


def run_algorithms(algorithms: list, settings: Settings,\
                                     env: Environment, outputDir: str) -> None:
    solutions = {}
    
    for algorithm in algorithms:
        initial_solution = Solution(mapping=env.mapping)
        logging.info(f'Run {algorithm}...')
        solver = registry[algorithm](env, settings)

        start_time = perf_counter()
        solution = solver.solve()
        solution.elapsed = perf_counter() - start_time
        solutions[algorithm] = solution

    for algorithm, solution in solutions.items():
        score = Score(env, settings)
        logging.info(f'[{algorithm}]')
        if solution.mapping is None or env.validate_mapping(solution.mapping):
            logging.info(f'  Objective: {score.repr_objective(solution)}')
            logging.info(f'  Active: {score.repr_savings_score(solution)}')
            logging.info(f'  Migration: {score.repr_migration_score(solution)}')
            logging.info(f'  Elapsed: {solution.elapsed:.4f}s')
            if solution.status is not None:
                logging.info(f'  Status: {solution.status}')
        else:
            logging.error(f'  INFEASIBLE SOLUTION')
        solution.save(outputDir + f'/{algorithm}-result.json')




def run_from_command_line() -> None:
    args = process_input_arguments()
    
    output_dir = f'runs/run-{get_time()}/'
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)   

    env = setup_enviroment(args, output_dir)
    settings = Settings(
            wa=args.wa,
            wm=args.wm,
            tl=args.tl,
            max_force_steps=args.f,
    )
    
    run_algorithms(args.algorithm, settings, env,  output_dir)
    

def run_example(problem_path: str, algorithm: str, time_limit:int =60, wa:float = 1, wm:int = 2):
    env = Environment.load(problem_path)

    settings = Settings(tl=time_limit, wa=wa, wm=wm)

    t_start = perf_counter()
    solution = algorithm(env,settings).solve()
    t = perf_counter() - t_start
    
    score = Score(env, settings)
    
    results = f"""
            Running time:                       {t} sec
            Objective function:                 {score.objective(solution)}
            Number of hosts in initial mapping: {score.savings_score(Solution(mapping=score.env.mapping))}
            Number of hosts in final mapping:   {score.savings_score(solution)}
            Number of released hosts:           {score.savings_score(Solution(mapping=score.env.mapping))-score.savings_score(solution)}
            Amount of migrated memory:          {score.migration_score(solution)} TiB\n"""
            
    print(results)

if __name__ == '__main__':
    
    registry = {
            'balcon': BalCon,
            'sercon-modified': SerCon,
            'sercon-original': SerConOriginal,
            'firstfit': FirstFit,
            'flowmodel': FlowModel,
            'flowmodel-relaxed': FlowModelRelaxed,
            'allocationmodel': AllocationModel,
    }
    
    run_from_command_line()
    #run_example(problem_path='../dataset/000.json', algorithm=registry['balcon'], time_limit=60, wa=1, wm=2)




