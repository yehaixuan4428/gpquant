import os
import random
from typing import Any

import numpy as np
import pandas as pd

from .Backtester import backtester_map
from .Fitness import fitness_map
from .Function import function_map
from .SyntaxTree import SyntaxTree
from tqdm import tqdm
from joblib import Parallel, delayed


class SymbolicRegressor:
    def __init__(
        self,
        population_size: int,
        tournament_size: int,
        generations: int,
        stopping_criteria: float,
        p_crossover: float,
        p_subtree_mutate: float,
        p_hoist_mutate: float,
        p_point_mutate: float,
        init_depth: tuple,
        init_method: str,
        function_set: list,
        variable_set: list,
        const_range: tuple,
        ts_const_range: tuple,
        build_preference: list,
        metric: str,
        transformer: str = None,
        transformer_kwargs: dict = None,
        parsimony_coefficient: float = 0,
        cache_dir: str = "./cache",
        pool_size: int = 1,
        best_n_children: int = 1,
        seed: int = 32
    ) -> None:
        self.seed = seed
        random.seed(seed)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir: str = cache_dir
        self.pool_size = pool_size
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.p_crossover = p_crossover
        self.p_subtree_mutate = p_subtree_mutate
        self.p_hoist_mutate = p_hoist_mutate
        self.p_point_mutate = p_point_mutate
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = (
            [function_map[f] for f in function_set]
            if function_set
            else list(function_map.values())
        )
        self.function_names = (
            function_set if function_set else list(function_map.keys())
        )

        self.variable_set = variable_set
        self.const_range = const_range
        self.ts_const_range = ts_const_range
        self.build_preference = build_preference
        self.metric = fitness_map[metric]
        self.transformer = None if transformer is None else backtester_map[transformer]
        self.transformer_kwargs = transformer_kwargs
        self.parsimony_coefficient = parsimony_coefficient
        self.trees: list[SyntaxTree] = []
        self.fitness: list[float] = []
        self.best_estimator: SyntaxTree = None
        self.best_fitness: float = None
        self.best_n_children: int = min(best_n_children, population_size)
        self.best_children_history: list[SyntaxTree] = []

    def __build(self) -> None:
        if self.pool_size > 1:

            def create_tree(
                id,
                init_depth,
                init_method,
                function_set,
                variable_set,
                const_range,
                ts_const_range,
                build_preference,
                metric,
                transformer,
                transformer_kwargs,
                parsimony_coefficient,
                cache_dir,
                seed
            ):
                return SyntaxTree(
                    id,
                    init_depth,
                    init_method,
                    function_set,
                    variable_set,
                    const_range,
                    ts_const_range,
                    build_preference,
                    metric,
                    transformer,
                    transformer_kwargs,
                    parsimony_coefficient,
                    cache_dir,
                    seed
                )

            self.trees = Parallel(n_jobs=self.pool_size)(
                delayed(create_tree)(
                    id,
                    self.init_depth,
                    self.init_method,
                    self.function_set,
                    self.variable_set,
                    self.const_range,
                    self.ts_const_range,
                    self.build_preference,
                    self.metric,
                    self.transformer,
                    self.transformer_kwargs,
                    self.parsimony_coefficient,
                    self.cache_dir,
                    self.seed
                )
                for id in range(self.population_size)
            )
        else:
            for i in range(self.population_size):
                self.trees.append(
                    SyntaxTree(
                        id=i,
                        init_depth=self.init_depth,
                        init_method=self.init_method,
                        function_set=self.function_set,
                        variable_set=self.variable_set,
                        const_range=self.const_range,
                        ts_const_range=self.ts_const_range,
                        build_preference=self.build_preference,
                        metric=self.metric,
                        transformer=self.transformer,
                        transformer_kwargs=self.transformer_kwargs,
                        parsimony_coefficient=self.parsimony_coefficient,
                        cache_dir=self.cache_dir,
                        seed = self.seed
                    )
                )

    def __tournament(self) -> SyntaxTree:
        contenders = random.sample(range(self.population_size), self.tournament_size)
        fitness = [self.fitness[i] for i in contenders]
        if self.metric.sign > 0:
            parent_index = contenders[np.nanargmax(fitness)]
        else:
            parent_index = contenders[np.nanargmin(fitness)]
        return self.trees[parent_index]

    def __evolve(self) -> None:
        offsprings = []
        method_probs = [
            self.p_crossover,
            self.p_subtree_mutate,
            self.p_hoist_mutate,
            self.p_point_mutate,
        ]
        method_probs = np.cumsum(method_probs)
        if method_probs[-1] > 1:
            raise ValueError(
                "sum of crossover and mutation probabilities should <= 1.0"
            )
        for _ in range(self.population_size):
            parent = self.__tournament()
            method = np.searchsorted(method_probs, random.random())
            if method == 0:
                # crossover
                donor = self.__tournament()
                offsprings.append(parent.crossover(donor))
            elif method == 1:
                # subtree mutation
                offsprings.append(parent.subtree_mutate())
            elif method == 2:
                # hoist mutation
                offsprings.append(parent.hoist_mutate())
            elif method == 3:
                # point mutation
                offsprings.append(parent.point_mutate())
            else:
                # reproduction
                offsprings.append(parent.reproduce())
        self.trees = offsprings

    def __log(self, i: int) -> None:
        print(f"------------Generation {str(i + 1).rjust(2)}------------")
        print(f"best estimator: {self.best_estimator}")
        print(f"best fitness: {self.best_fitness}")
        if self.last_best_fitness is not None and self.last_best_fitness is not np.nan:
            error = np.abs(
                (self.best_fitness - self.last_best_fitness) / self.last_best_fitness
            )
            print(f"best fitness improve: {error:.3%}")

    def fit(self, X: pd.DataFrame, y: pd.Series, is_cached: bool = False) -> None:
        self.last_best_fitness = None
        print("Build init trees...")
        self.__build()
        for i in range(self.generations):
            print(f"Calculate generation {i+1}...")

            if self.pool_size == 1:
                self.fitness = np.array(
                    [tree.fitness(X, y, is_cached) for tree in tqdm(self.trees)]
                )
            else:
                t = tqdm(self.trees)

                # def func(tree_func, X, y, tree_describ):
                #     print(tree_describ)
                #     return tree_func(X, y)

                self.fitness = Parallel(n_jobs=self.pool_size)(
                    delayed(tree.fitness)(X, y, is_cached)
                    for tree in t
                    # delayed(func)(tree.fitness, X, y, str(tree.nodes[0]))
                    # for tree in t
                )

            print("Generation fitnesses are calculated...")

            self.best_estimator = self.trees[
                np.nanargmax(self.metric.sign * self.fitness)
            ]
            self.best_fitness = self.metric.sign * np.nanmax(
                self.metric.sign * self.fitness
            )
            self.__log(i)
            # save the best n children into records
            top_n = np.argsort(self.fitness)[-self.best_n_children :]
            top_trees = [self.trees[i] for i in top_n]
            self.best_children_history.append(top_trees)

            if (
                self.last_best_fitness is not None
                and self.last_best_fitness is not np.nan
            ):
                if (
                    np.abs(
                        (self.best_fitness - self.last_best_fitness)
                        / self.last_best_fitness
                    )
                    < self.stopping_criteria
                ):
                    break
            self.last_best_fitness = self.best_fitness
            # if self.metric.sign * (self.best_fitness - self.stopping_criteria) > 0:
            #     break
            self.__evolve()

    def predict(self, X: pd.DataFrame) -> Any:
        return self.best_estimator.execute(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.best_estimator is None:
            raise AttributeError("cannot call score without `self.best_estimator`")
        return self.best_estimator.fitness(X, y)
