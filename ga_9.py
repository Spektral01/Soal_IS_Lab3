from deap import base, algorithms
from deap import creator
from deap import tools

import algelitism

import random
import matplotlib.pyplot as plt
import numpy as np

LOW, UP = -6, 7
ETA = 20
LENGTH_CHROM = 2    # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 50   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.2        # вероятность мутации индивидуума
MAX_GENERATIONS = 50    # максимальное количество поколений
HALL_OF_FAME_SIZE = 5

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #класс описывает значения приспособленности особи
creator.create("Individual", list, fitness=creator.FitnessMin) #класс представляющий саму особь


def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]


toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint, LOW, UP) #генерация рандомных точек
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint) #создание особи
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator) #создание популяции

#вычисление приспособленности особи
population = toolbox.populationCreator(n=POPULATION_SIZE)


def porabola(individual):
    x, y = individual
    f = (x*x)+4
    return f,

toolbox.register("evaluate", porabola) #вычисление приспособленности особи
toolbox.register("select", tools.selTournament, tournsize=3) #отбор особи
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA) #функция для скрещивания  
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM) #функция определяет мутации

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

#график
import time
def show(ax, xgrid, ygrid, f):
    ptMins = [[0.0, 4.0]]



    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
    ax.scatter(*zip(*population), color='green', s=2, zorder=0)
 

    plt.draw()
    plt.gcf().canvas.flush_events()
    

    time.sleep(0.2)


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xgrid, ygrid = np.meshgrid(x, y)

   

f_porabola = xgrid**2  + 4 - ygrid



plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)


ax.set_xlim(LOW+3, UP-3)
ax.set_ylim(LOW+3, UP-3)



#algelitism.eaSimpleElitism
#algorithms.eaSimple
population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        callback=(show, (ax, xgrid, ygrid, f_porabola)),
                                        verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]
print(best)


 
plt.ioff()
plt.show()

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
