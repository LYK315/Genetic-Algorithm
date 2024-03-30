import time
import random
import matplotlib.pyplot as plt
import math
from SteadyStateClass import SSAGeneral

##################################
# Reusable (Run Steady State GA) #
##################################
def runGA():
    """
    Notes:
    1. This function executes Steady State Genetic Algorithm (SSGA).
    2. Fitness function and Initializ Population are designed to be overriden to be able to execute Problem 1 / Problem 3 respectively.
    
    SSGA Steps:
    1. Initialize Population
    2. Calculate Fitness Function of each Individual in Population
    3. Crossover / Clone to get Offspring
    4. Mutate Offspring (if cointoss pass)
    5. Replace worst individual with Offspring in Population
    ** Repeat 2 - 5 untill max generation is reached, then:
    6. Plot Graph (only for Problem 1 - ASCII Art)
    7. Print Summary of GA Run

    Return Arguments:
    1. Best Chromosome
    """
    start_time = time.time()
    Population = []
    AllFits = [0] * PopulationSize
    MaxFits = []
    AvgFits = []
    FitnessEvalCount = 0

    # Initialize Population
    initialize_population(Population)

    # Evaluate and store individual fitness separately, this minimizes compuatation cost by only evaluating fitness of the new offspring in each generation
    for Index, Individual in enumerate (Population):
        # Different Fitness Function is used for Problem 1 & Problem 3
        if FitnessFunction is None:
            AllFits[Index] = SSGA_Class.check_fitness(Target, Individual)
        else:
            AllFits[Index] = FitnessFunction(Individual)
        FitnessEvalCount += 1
    
    # Evaluate Generations
    for GenerationCount in range (MaxGeneration):
        track_fitness(AllFits, AvgFits, MaxFits)

        # Dont print progress when analysing hyper-parameter combination
        display_progress(GenerationCount, AvgFits, MaxFits) if not hyperAnalysis else None
        
        # Select Parents for Crossover / Clone
        if SSGA_Class.cointoss(CrossOverRate):
            Parent1 = select_method(SelectionMethod, Population, AllFits)
            Parent2 = select_method(SelectionMethod, Population, AllFits)
            Offspring = SSGA_Class.crossover(Parent1, Parent2)
        else:
            Offspring = select_method(SelectionMethod, Population, AllFits)
        
        # Mutate Offspring
        Offspring = SSGA_Class.mutate(Offspring, MutationRate)

        # Replcae worst individual with Offspring
        ElderIndex = SSGA_Class.select_worst(AllFits, Reverse=False)
        Population[ElderIndex] = Offspring

        # Evaluate fitness of the new Offspring
        if FitnessFunction is None:
            AllFits[ElderIndex] = SSGA_Class.check_fitness(Target, Offspring)
        else:
            AllFits[ElderIndex] = FitnessFunction(Offspring)
        FitnessEvalCount += 1

        # Update Best Chromosome Details
        global OptimaIndex, BestIndex, SolutionIndex, BestFitness
        for Index, Score in enumerate (AllFits):
            # Index of Solution (if found)
            if FitnessFunction is None and SolutionIndex is None:
                if (TargetFit == Score):
                    SolutionIndex = GenerationCount+1
            # Index of Current Best Chromosome
            if (Score > BestFitness):
                BestFitness = Score
                OptimaIndex = Index
                BestIndex = GenerationCount+1
    
    end_time = time.time()

    # Dont plot graph for Problem 3 & while analysing hyper-parameters
    if FitnessFunction is None and not hyperAnalysis:
        plot_fitness_generation(GenerationCount+1, MaxFits, AvgFits)

    # Print Summary of GA Run
    summary_msg(Population, FitnessEvalCount, AvgFits, start_time, end_time)

    # Return the BEST Chromosome in Population
    BestChromosome = Population[SSGA_Class.select_worst(AllFits, Reverse=True)]
    return BestChromosome



#########################
# Problem 1 - ASCII-Art #
#########################
def run_Ascii():
    """
    This function use "runGA()" to solve Problem 1 - ASCII Art.

    Representation:
    - Binary

    Decoded As:
    - ASCII-Art Image

    Initialize Population:
    - Generate random individuals (bitstring) in population list.

    Selection Method (select individuals from population):
    - Tournament / FPS

    Fitness Function: 
    - fitness = sum( Individual[i] == Target[i] forall i in Individual )
    - Maximizing function

    Evaluation:
    - Generate Offspring by Crossover / Cloning / Mutatation.
    - Replace worst individual in population with Offspring.

    Termination Criteria:
    - When Max Generation is Reached.
    """
    global PopulationSize, NBits, MaxGeneration
    global CrossOverRate, MutationRate, TournamentSize
    global FitnessFunction, Target, TargetFit
    global SolutionIndex, OptimaIndex
    global hyperAnalysis

    PopulationSize = 20
    NBits = 85
    MaxGeneration = 1500
    CrossOverRate = 0.75
    MutationRate = 0.01 # Each Bit
    TournamentSize = 2
    FitnessFunction = None
    SolutionIndex = None
    OptimaIndex = None
    Target = "0001111000001100000100001000101100010000000010001100010011100111111000011101001000010"
    TargetFit = NBits
    hyperAnalysis = False

    runGA()



#################################
# Problem 3 - Fitness Functions #
#################################
def decodeRealValue(Binary, minVal, maxVal):
    """
    Convert binary string of X and Y to Real Value.
    """ 
    decimal = int(Binary, 2)
    return minVal + (decimal / (2**BitsPerNum - 1)) * (maxVal - minVal)

def decodeChromosome (Chromosome, x_val, y_val):
    """
    Seperate Chromosome into to binary strings, X and Y.
    """
    x_binary = Chromosome[:BitsPerNum]
    y_binary = Chromosome[BitsPerNum:]

    x = decodeRealValue(x_binary, x_val[0], x_val[1])
    y = decodeRealValue(y_binary, y_val[0], y_val[1])

    return x, y

# Ackley Function (Single-objective)
def ackley_function(Chromosome):
    """
    Fitness Function algorithm for Ackley. Do not change any thing here.
    * x_val and y_val contains Min and Max value for X and Y.
    """
    Padding = 1000

    # Min & Max value of X & Y
    x_val = [-5, 5]
    y_val = [-5, 5]
    x, y = decodeChromosome(Chromosome, x_val, y_val)

    score = -20 * math.exp(-0.2 * math.sqrt(0.5*(x*x + y*y))) - math.exp(0.5*(math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.exp(1) + 20

    if MinimiseFunction:
        return Padding - score
    else:
        return score

# Rosenbrock Function (Constrained with a Cubic and a Line)
def rosenbrock_function(Chromosome):
    """
    Fitness Function algorithm for Rosenbrock. Do not change any thing here.
    * x_val and y_val contains Min and Max value for X and Y.
    """
    Padding = 1000

    # Min & Max value of X & Y
    x_val = [-1.5, 1.5]
    y_val = [-0.5, 2.5]
    x, y = decodeChromosome(Chromosome, x_val, y_val)

    if ((x - 1)**3 - y > 0) or (x + y - 2 > 0):
        return 1
    
    score = (1 - x)**2 + 100*(y - x*x)**2

    if MinimiseFunction:
        return Padding - score
    else:
        return score

# Run Ackley Function
def run_Ackley(PopSize, MaxGen, Minimise):
    """
    Set Hyper-parameters for running Ackley Function here.
    """
    global PopulationSize, MaxGeneration, NBits, BitsPerNum
    global CrossOverRate, MutationRate, TournamentSize
    global FitnessFunction, MinimiseFunction, displayASCII

    # Set Hyper-Parameters Here
    PopulationSize = PopSize
    MaxGeneration = MaxGen
    BitsPerNum = 50
    TotalNums = 2
    NBits = BitsPerNum * TotalNums
    CrossOverRate = 1
    MutationRate = 0.02
    TournamentSize = 4
    FitnessFunction = ackley_function
    MinimiseFunction = Minimise
    displayASCII = False
    Padding = 1000
    x_val = [-5, 5]
    y_val = [-5, 5]

    # Run Fitness Function
    run_FitnessFunction(Padding, x_val, y_val)

# Run Rosenbrock Function
def run_Rosenbrock(PopSize, MaxGen, Minimise):
    """
    Set Hyper-parameters for running Rosenbrock Constrained Function here.
    """
    global PopulationSize, MaxGeneration, NBits, BitsPerNum
    global CrossOverRate, MutationRate, TournamentSize
    global FitnessFunction, MinimiseFunction, displayASCII

    # Set Hyper-Parameters Here
    PopulationSize = PopSize
    MaxGeneration = MaxGen
    BitsPerNum = 50
    TotalNums = 2
    NBits = BitsPerNum * TotalNums
    CrossOverRate = 1
    MutationRate = 0.1
    TournamentSize = 2
    FitnessFunction = rosenbrock_function
    MinimiseFunction = Minimise
    displayASCII = False
    Padding = 1000
    x_val = [-1.5, 1.5]
    y_val = [-0.5, 2.5]

    # Run Fitness Function
    run_FitnessFunction(Padding, x_val, y_val)

# Run Fitness Functions
def run_FitnessFunction(Padding, x_val, y_val):
    """
    This function simulates Problem 3 - Fitness Functions (Ackley / Rosenbock Constrained) 
    by overriding the Fitness Function and Initialize Population in "runGA()".

    Notes:
    - Nothing need to be changed here. Just change everything in "boilerplate" or 
      "run_Ackley" / "run_Rosenbrock" if you wish to ammend the Hyper-paramaters.

    Representation:
    - Binary

    Decoded As:
    - Real Value

    Initialize Population:
    - Generate random individuals (bitstring) in population list.

    Selection Method (select individuals from population):
    - Tournament

    Simulate Fitness Function: 
    - Ackley Function
    - Rosenbrock Function (Constrained with a Cubic and a Line)

    Evaluation:
    - Generate Offspring by Crossover / Cloning / Mutatation.
    - Replace worst individual in population with Offspring.

    Termination Criteria:
    - When Max Generation is Reached.
    """

    # Run GA and get Best Chromosome as Return
    Chromosome = runGA()

    # Print summary message
    if FitnessFunction == ackley_function:
        print("\n- Ackley Function (Single-Objective) Results -")
        print("Resolution: ", (5 - -5)/(2**NBits - 1), "(x and y)")
    else:
        print("\n- Rosenbrock Function (Constrained) Results -")
        print("Resolution: ", (1.5 - -1.5)/(2**NBits - 1), "(x), ", (2.5 - -0.5)/(2**NBits - 1), "(y)")
    print("Minimize  : ", MinimiseFunction)
    print("Best Solution: Generation", BestIndex)
    print("Best Chromosome: ", Chromosome)
    print("Fraction Search Space Explored: ", (MaxGeneration*PopulationSize) / (NBits**NBits))

    # Get score of best chromosome
    score = FitnessFunction(Chromosome)

    if MinimiseFunction:
        # Remove the extra padding from the final result if minimizing
        score = score - Padding

    # Print summary message, cont.
    print("Best Score: ", score)
    x, y = decodeChromosome(Chromosome, x_val, y_val)
    print('X: ', x)
    print('Y: ', y)

    print('\nEnd of Optimisation Function.\n')

# Print Resolution
def print_resolution():
    """
    Nothing important, just to print the resolution of Fitness Function.
    Equation: range / Number ofintervals
    """
    print("Ackley function (x and y): ", (5 - -5)/(2**50 - 1))
    print("Rosenbrock function (x)  : ", (1.5 - -1.5)/(2**50 - 1))
    print("Rosenbrock function (y)  : ", (2.5 - -0.5)/(2**50 - 1))



#######################################
# Global Variables / Hyper-Parameters #
#######################################
"""General GA Variables"""
PopulationSize = 0
NBits = 0
MaxGeneration = 0
CrossOverRate = 0
MutationRate = 0
SelectionMethod = "Tournament" # Tournament / FPS 
AcceptanceProb = 0.9
TournamentSize = 0

"""Specific for Problem 3"""
SolutionIndex = None
OptimaIndex = None
BestIndex = None
Target = None
TargetFit = NBits

"""Other Helper Variablers"""
displayASCII = True
hyperAnalysis = False
FitnessFunction = None
BestFitness = 0



########################
# Steady-Sate GA Class #
########################
class SSGA_Class(SSAGeneral):
    """
    Contains overrided SSGA functions to solve Problem 1 & Problem 3

    Note:
    - decode() here is only for Problem 1
    - Problem 3's decode function is defined under "Problem 3 - Fitness Functions" section
    """
    def check_fitness(Target, Individual):
        """
        Fitness Function.
        fitness = sum( Individual[i] == Target[i] forall i in Individual )
        """
        return sum(Target[i] == bit for i, bit in enumerate(Individual))

    def crossover(Parent1, Parent2):
        ChiasmaLocation = random.randint(1, len(Parent1)-1)
        Chromosome = Parent1[:ChiasmaLocation] + Parent2[ChiasmaLocation:]
        return Chromosome

    def mutate(Offspring, MutationRate):
        Mutant = ""
        for Gene in Offspring:
            if (SSAGeneral.cointoss(MutationRate)):
                Mutant += "1" if Gene == "0" else "0"
            else:
                Mutant += Gene

        return Mutant

    def decode(Genotype):
        """Convert chromosome's genotype to phenotype"""
        Phenotype = ''

        for index, gene in enumerate(Genotype):
            Phenotype += ''.join(str(gene))
            if index == 67 or index == 50 or index == 33 or index == 16:
                Phenotype += '\n'

        return Phenotype



############################
# General Helper Functions #
############################
def initialize_population(Population):
    """
    Generate initial population (binary strings) for the problem.
    """
    def generate_individual():
        # Generate random individuals (chromosome)
        return ''.join(str(random.randint(0, 1)) for i in range(NBits))

    Population[:] = [generate_individual() for i in range (PopulationSize)]

def display_progress(GenerationCount, AvgFits, MaxFits):
    """
    Print current progress of GA. (Generation Count, Average Fitness, Max Fitness)
    """
    print("Generation:{:<5d}  Avg. Fitness: {:<20}  Best Fitness: {:<20}".format(GenerationCount, AvgFits[GenerationCount], MaxFits[GenerationCount]))

def track_fitness(AllFits, AvgFits, MaxFits):
  """
  Track fitness values of each generation, will be usefull when plotting histogram.
  """
  MaxFits.append(max(AllFits))
  AvgFits.append(sum(AllFits) / len(AllFits))

def select_method(Method, Population, AllFits):
    """
    Invoke selection method from "Steady State Class Package".
    """
    if Method == 'Tournament':
        return SSGA_Class.select_tournament(Population, AllFits, AcceptanceProb, TournamentSize)
    elif Method == 'FPS':
        return SSGA_Class.select_fps(Population, AllFits)
    else:
        print("Wrong Selection Method Provided !")

def plot_fitness_generation(GenerationCount, MaxFits, AvgFits):
    """
    Plots histogram showing the Max Fitness and Average Fitness througout evaluation. (For Problem 1 - ASCII ART Only)
    """
    plt.plot(range(GenerationCount), MaxFits, label="Max")
    plt.plot(range(GenerationCount), AvgFits, label="Avg") 
    plt.legend()
    plt.title("Avg vs Max Fitness per Generation")
    plt.show(block=True)

def summary_msg(Population, FitnessEvalCount, AvgFits, start, end):
    """
    Prints a summary message on terminal at the end of program. Hyper-paramaters, Execution Details, Best Solution will be printed.
    """
    total_avg = 0
    for avg in AvgFits:
        total_avg += avg

    print("\n[Results]")

    print("- Hyper-Parameters -")
    print("Population Size: ", PopulationSize)
    print("Generation Size: ", MaxGeneration)
    print(f"Mutation Rate: {MutationRate:.2f}")
    print(f"Crossover Rate : {CrossOverRate:.2f}")

    print("\n- Execution Details -")
    print("Solutions Sampled: ", PopulationSize * MaxGeneration)
    print("Execution Time   : ", end - start, "(s)")
    print(f"Selection Method :  {SelectionMethod} (Select {TournamentSize})")
    print("Fitness Evaluations: ", FitnessEvalCount)
    print("Total Avg Fitness  : ", total_avg/len(AvgFits))


    # Print different summary message for Problem 1 & Problem 3
    if FitnessFunction is None:
        if SolutionIndex is not None:
            print(f"\nASCII Art Solution REACHED at Generation {SolutionIndex}.")
            if displayASCII:
                print(SSGA_Class.decode(Population[OptimaIndex]))
        else:
            print(f"\nASCII Art Solution NOT FOUND within {MaxGeneration} generations. Try Again.")
        print("\nEnd of Program.")
    else:
        print("Best Fitness  :", BestFitness)
        print("Reached At Gen:", BestIndex)



######################
# Python Boilerplate #
######################
if __name__ == "__main__":
    """
    Run all Codes here, uncomment the code you wish to run and comment the others. The codes you can run are:
    1. Problem 1 - ASCII ART Optimisation Problem
    2. Problem 3 - Simulate Fitness Functions
        - Ackley Function (Minimise / Maximise)
        - Rosenbrock Function Constrained with a Cubic and a Line (Minimise / Maximise)
        * Pass in the Population Size & Max Generations to run.
    
    Notes:
    1. You may choose to use Tournament/FPS for ASCII-Art (Problem 1). Ammend under "General GA Variables"
    2. Tournament only for Problem 3, FPS is not tested here.
    """

    """Problem 1 - ASCII ART"""
    # run_Ascii()

    """Problem 3 - Fitness Functions"""
    # run_Ackley(50, 5000, Minimise=True) # Minimise
    # run_Ackley(50, 5000, Minimise=False) # Maximise
    run_Rosenbrock(50, 40000, Minimise=True) # Minimize
    # run_Rosenbrock(50, 10000, Minimise=False) # Maximise

    """Just To Check Resolution of Fitness Function"""
    # print_resolution()
