# select k genes from current population
# population is a list of chromosomes
import operator
import random

class SelectionMethod:

	def __init__(self,sample_size):
		self.k = sample_size

	def truncationSelection(self, population):
		population.sort(key=operator.attrgetter('Fitness'), reverse=True)
		return population[:self.k]

	def tournamentSelection(self, population):
		# select best out of k randomly chosen, k times
		new = []
		for i in range(self.k):
			sample = random.sample(population, self.k)
			best = sample[0]
			for element in sample[1:]:
				if element.Fitness > best.Fitness:
					best = element
			new.append(best)
		return new

	# def RouletteWheelSelection(self, population):
	# y = maximum(y) .- y
	# cat = Categorical(normalize(y, 1))
	# return [rand(cat, 2) for i in y]
