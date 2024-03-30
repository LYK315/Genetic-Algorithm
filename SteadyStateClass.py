import random
from abc import ABC, abstractmethod
import more_itertools

########################
# GA Selection Methods #
########################
class SelectionMethod():
    """
    Contains two common selection method in this class:
    - Tournamenet Selection
    - Fitness Proportional Selection (FPS)

    * Designed as the default selection methods for Steady State Genetic Algorithm.
    """
    def select_tournament(Population, AllFitness, AcceptanceProb, TournamentSize, Reverse=True):
        """
        Tournamanet Selection 
        """
        Tournament = [()] * TournamentSize

        for i in range(TournamentSize):
            rndIndex = random.randint(0, len(Population) - 1)
            Tournament[i] = (Population[rndIndex], AllFitness[rndIndex])

        Tournament.sort(key=lambda Indiv: Indiv[1], reverse=Reverse)
        
        for Individual in Tournament:
            if SSAGeneral.cointoss(AcceptanceProb):
                return Individual[0]
            else:
                return Tournament[-1][0]

    def select_fps(Population, AllFitness, Reverse=True):
        """
        Fitness Proportional Selection Selection (FPS)
        """
        FitnessScores = AllFitness

        MinFitScore = min(FitnessScores)
        MaxFitnessScore = max(FitnessScores)

        # Add a small number in case all value is zero due to convergence.
        FitnessScores = [0.1 + i - MinFitScore for i in FitnessScores]

        if Reverse == False:
            # Add a small number to make max solutions still be a tiny bit representable.
            FitnessScores = [0.1 + MaxFitnessScore - i for i in FitnessScores]

        LeftBoundary = [sum(FitnessScores[:i]) for i in range(len(Population))]
        RightBoundary = [sum(FitnessScores[:i+1])for i in range(len(Population))]

        RandomIndex = sum(FitnessScores) * random.random()
        SelectedIndex = more_itertools.first_true(zip(range(len(Population)), LeftBoundary, RightBoundary), pred=lambda T: T[1] <= RandomIndex < T[2])[0]

        return Population[SelectedIndex]


##########################
# General SS GA Features #
##########################
class SSAGeneral(ABC, SelectionMethod):
    """
    Contains general features of Steady State Genetic Algorithm (SSGA). 
    
    Case specific features are designed as abstract method so for user to override their own algorithm to sovle different problems:
    - Fitness Function
    - Crossover
    - Mutate
    - Decode
    """
    def cointoss(Bias):
        return random.random() < Bias

    def select_worst(Allfits, Reverse=False):
        """
        Select the worst individual in population. 
        - Select by Minimum Fitness Score by default
        - Select by Maximum if Reverse = True
        """
        if Reverse:
            return (Allfits.index(max(Allfits)))
        else:
            return (Allfits.index(min(Allfits)))

    @abstractmethod
    def check_fitness(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def decode(self):
        pass
