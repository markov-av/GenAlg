# -*- coding: utf-8 -*-
import numpy as np
import random


class Solver_8_queens:
    '''
    Dummy constructor representing proper interface
    '''

    def __init__(self, pop_size=200, cross_prob=0.11, mut_prob=0.2):
        self.pool_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

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
            general_population = self.mutation(self.crossing(pool=pool))
            population = general_population + pool
            MainPool = self.selection(population=population)
            if len(MainPool) == 1:
                break
            pool = MainPool[:self.pool_size]
            epoch_num += 1
            print(MainPool)
        visualization = self.visualization(MainPool[0])
        return best_fit, epoch_num, visualization

    def get_pool(self, pool_size):
        pool = []
        lst = [x for x in range(8)]
        for i in range(pool_size):
            np.random.shuffle(lst)
            pool.append([*lst])
        return pool

    def crossing(self, pool):
        general_population = []
        for _ in range(self.pool_size):
            a = random.randint(0, len(pool) - 1)
            b = random.randint(0, len(pool) - 1)
            pool_gen = []
            for i in range(len(pool[a])):
                ran_1 = random.randint(0, 100)
                ran_2 = random.randint(0, 100)
                if ran_1 >= ran_2:
                    if pool[a][i] in pool_gen:
                        j = pool[b][i]
                    else:
                        j = pool[a][i]
                    pool_gen.append(j)
                if ran_1 < ran_2:
                    if pool[b][i] in pool_gen:
                        j = pool[a][i]
                    else:
                        j = pool[b][i]
                    pool_gen.append(j)
                if len(pool_gen) == 8:
                    general_population.append([*pool_gen])
        return general_population

    # Мутация
    def mutation(self, general_population):
        for i in range(len(general_population)):
            if random.random() < self.mut_prob:
                general_population[i][random.randint(0, 7)] == random.randint(0, 7)
        return general_population

    # population = pool+general_population
    def get_fit(self, lst):
        k = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(i - j) == abs(lst[i] - lst[j]):
                    k += 1
        if len(lst) != len(set(lst)):
            k += 8 - len(set(lst))

        return (1 - k/8)

    # TODO: нужно переписать селекцию, убрать ифы
    # Отбор
    def selection(self, population):
        k = 0
        pop_gen_pool = []
        pop_gen_pool_high = []
        pop_gen_pool_mid = []
        pop_gen_pool_low = []
        for lst in population:
            for i in range(8):
                for j in range(i + 1, 8):
                    if abs(i - j) == abs(lst[i] - lst[j]):
                        k += 1
            if len(lst) != len(set(lst)):
                k += 8 - len(set(lst))
            if len(pop_gen_pool) + len(pop_gen_pool_high) + len(pop_gen_pool_mid) + len(
                    pop_gen_pool_low) < self.pool_size:
                if (1 - k / 8) == 1:
                    pop_gen_pool.append(lst)
                    return [lst]
                if (1 - k / 8) < 1 and (1 - k / 8) >= 0.75:
                    pop_gen_pool_high.append(lst)
                if (1 - k / 8) < 0.75 and (1 - k / 8) >= 0.5:
                    pop_gen_pool_mid.append(lst)
                if (1 - k / 8) < 0.5:
                    pop_gen_pool_low.append(lst)
            k = 0
        MainPool = pop_gen_pool + pop_gen_pool_high + pop_gen_pool_mid + pop_gen_pool_low
        return MainPool

    # TODO: переписать функцию, чтобы она не принтела результат, а возвращала строчку
    def visualization(self, lstMain):
        visual = ''
        for p in range(8):
            visual += lstMain[p] * ' * ' + ' Q ' + ' * ' * (7 - lstMain[p]) + '\n'
        return visual


solver = Solver_8_queens()
best_fit, epoch_num, visualization = solver.solve()
print("Best solution:")
print("Fitness:", best_fit)
print("Iterations:", epoch_num)
print(visualization)
