# -*- coding: utf-8 -*-
import numpy as np
import random


class Solver_8_queens:
    '''
    Dummy constructor representing proper interface
    '''

    def __init__(self, pop_size=100, cross_prob=0.85, mut_prob=0.5, min_fitness=1):
        self.pool_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.min_fitness = min_fitness

    '''
    Dummy method representing proper interface
    '''

    # TODO: добавить min_fitness
    def solve(self, min_fitness=0.9, max_epochs=2500):
        best_fit = None
        epoch_num = None
        visualization = None
        population = self.get_pool(pool_size=self.pool_size)
        epoch_num = 0
        while epoch_num <= max_epochs:
            spinogryzy = self.mutation(self.crossing(self.chat_rullet(population=population)))
            top_population = self.reduce(population=spinogryzy + population)
            if len(top_population) == 1:
                break
            population = top_population
            epoch_num += 1
        visualization = self.visualization(top_population[0])
        best_fit = self.get_fit(top_population[0])
        return best_fit, epoch_num, visualization

    def get_pool(self, pool_size):
        population = []
        lst = ['000', '001', '010', '011', '100', '101', '110', '111']
        for i in range(pool_size):
            np.random.shuffle(lst)
            population.append(''.join(lst))
        return population

    def crossing(self, population):
        crossing_population = []
        for i in range(0, len(population) - 1, 2):
            k = random.randint(1, 23)
            if random.random() < self.cross_prob:
                child_one = population[i][:k] + population[i + 1][k:]
                child_two = population[i + 1][:k] + population[i][k:]
                crossing_population.append(child_one)
                crossing_population.append(child_two)
            else:
                crossing_population.append(population[i])
                crossing_population.append(population[i+1])
        return crossing_population

    def chat_rullet(self, population):
        best_fit_pop = []
        sum_fit = 0
        for j in range(len(population)):
            sum_fit += self.get_fit(population[j])
        probs = []
        for j in range(self.pool_size):
            probs.append(self.get_fit(population[j]) / sum_fit)
        rulet = [probs[0]]
        for i in range(1, len(probs)):
            rulet.append(rulet[-1] + probs[i])
        for _ in range(self.pool_size):
            k = random.random()
            for i, value in enumerate(rulet):
                if k < value:
                    best_fit_pop.append(population[i])
                    break
        return best_fit_pop

    # Мутация
    def mutation(self, population):
        for i in range(len(population)):
            if random.random() < self.mut_prob:
                k = random.randint(0, 23)
                if population[i][k] == '0':
                    population[i] = population[i][:k] + '1' + population[i][k + 1:]
                else:
                    population[i] = population[i][:k] + '0' + population[i][k + 1:]
        return population

    def decoding(self, individ: str) -> list:
        lst = []
        for i in range(0, len(individ), 3):
            lst.append(int(individ[i: i + 3], 2))
        return lst

    def get_fit(self, lst: str):
        k = 0
        lst = self.decoding(lst)
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(i - j) == abs(lst[i] - lst[j]):
                    k += 1
        if len(lst) != len(set(lst)):
            k += 8 - len(set(lst))
        return (1 - k / 28)
        # Число 28, максимально возможное количество конфликтов на доске

    # Отбор
    def reduce(self, population):
        sorted_population = sorted(population, key=self.get_fit, reverse=True)
        if self.get_fit(sorted_population[0]) == 1:
            return [sorted_population[0]]
        if self.get_fit(sorted_population[0]) > self.min_fitness:
            return [sorted_population[0]]
        return sorted_population[:self.pool_size]

    def visualization(self, individ):
        visual = ''
        individ = self.decoding(individ)
        for p in range(8):
            visual += individ[p] * ' * ' + ' Q ' + ' * ' * (7 - individ[p]) + '\n'
        return visual


solver = Solver_8_queens()
best_fit, epoch_num, visualization = solver.solve()
print("Best solution:")
print("Fitness:", best_fit)
print("Iterations:", epoch_num)
print(visualization)
