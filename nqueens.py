# -*- coding: utf-8 -*-
import numpy as np
import random


class Solver_8_queens:
    '''
    Dummy constructor representing proper interface
    '''

    def __init__(self, pop_size=40, cross_prob=0.85, mut_prob=0.25, min_fitness=1):
        self.pool_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.min_fitness = min_fitness

    '''
    Dummy method representing proper interface
    '''

    # TODO: добавить min_fitness
    def solve(self, min_fitness=0.9, max_epochs=1000):
        best_fit = None
        epoch_num = None
        visualization = None
        pool = self.get_pool(pool_size=self.pool_size)
        epoch_num = 0
        while epoch_num <= max_epochs:
            general_population = self.mutation(self.crossing(self.chat_rullet(pool=pool)))
            population = general_population + pool
            MainPool = self.selection(population=population)
            if len(MainPool) == 1:
                break
            pool = MainPool
            epoch_num += 1
        visualization = self.visualization(MainPool[0])
        best_fit = self.get_fit(MainPool[0])
        return best_fit, epoch_num, visualization

    def get_pool(self, pool_size):
        pool = []
        lst = ['000', '001', '010', '011', '100', '101', '110', '111']
        for i in range(pool_size):
            np.random.shuffle(lst)
            pool.append(''.join(lst))
        return pool

    def crossing(self, pool):
        general_population = []
        for i in range(0, len(pool) - 1, 2):
            k = random.randint(1, 23)
            if random.random() < self.cross_prob:
                child_one = pool[i][:k] + pool[i + 1][k:]
                child_two = pool[i + 1][:k] + pool[i][k:]
                general_population.append(child_one)
                general_population.append(child_two)
            else:
                general_population.append(pool[i])
                general_population.append(pool[i+1])
        return general_population

    def chat_rullet(self, pool):
        best_fit_pop = []
        sum_fit = 0
        for j in range(len(pool)):
            sum_fit += self.get_fit(pool[j])
        probs = []
        for jj in range(self.pool_size):
            probs.append(self.get_fit(pool[jj]) / sum_fit)
        rulet = [probs[0]]
        for i in range(1, len(probs)):
            rulet.append(rulet[-1] + probs[i])
        for _ in range(self.pool_size):
            k = random.random()
            for i, value in enumerate(rulet):
                if k < value:
                    best_fit_pop.append(pool[i])
                    break
        return best_fit_pop

    # Мутация
    def mutation(self, general_population):
        for i in range(len(general_population)):
            if random.random() < self.mut_prob:
                k = random.randint(1, 23)
                if general_population[i][k] == '0':
                    general_population[i] = general_population[i][:k] + '1' + general_population[i][k + 1:]
                else:
                    general_population[i] = general_population[i][:k] + '0' + general_population[i][k + 1:]
        return general_population

    def decoding(self, individ: str) -> list:
        lst = []
        for i in range(0, len(individ), 3):
            #print(individ, individ[i: i + 3])
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
    def selection(self, population):
        MainPool = sorted(population, key=self.get_fit, reverse=True)
        if self.get_fit(MainPool[0]) == 1:
            return [MainPool[0]]
        if self.get_fit(MainPool[0]) > self.min_fitness:
            return [MainPool[0]]
        return MainPool[:self.pool_size]

    def visualization(self, lstMain):
        visual = ''
        lstMain = self.decoding(lstMain)
        for p in range(8):
            visual += lstMain[p] * ' * ' + ' Q ' + ' * ' * (7 - lstMain[p]) + '\n'
        return visual


solver = Solver_8_queens()
best_fit, epoch_num, visualization = solver.solve()
print("Best solution:")
print("Fitness:", best_fit)
print("Iterations:", epoch_num)
print(visualization)
