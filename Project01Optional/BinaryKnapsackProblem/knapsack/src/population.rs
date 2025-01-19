use crate::individual::Individual;
use crate::individual::Item;
use rand::Rng;

// Constant params to configure the enviroment ect
use crate::params::{KS_CAPACITY, CROSSOVER_POINT, MUTATION_RATE};


/**
 * Population struct for the knapsack problem
 * individuals: vector of individuals in the population
 * size: size of the population
 */
#[derive(Debug, Clone)]
pub struct Population {
    pub individuals: Vec<Individual>, 
    pub size: usize,                 
}

/**
 * Implementation of the Population struct
 */
impl Population {
    /**
     * Constructor for Population which generates a new population of individuals
     * size: size of the population
     */
    pub fn new(size: usize) -> Self {
        let individuals = (0..size)
            .map(|_| Individual::new()) 
            .collect();
        Population { individuals, size }
    }

    /**
     * Evaluate the fitness of the population based on the items in the knapsack
     * items: vector of items in the knapsack
     */
    pub fn evaluate_fitness(&mut self, items: &[Item]) {
        for individual in &mut self.individuals {
            individual.calculate_fitness(items);
        }
    }

    /**
     * Perform crossover between two parents to create two children
     * parent1: reference to the first parent
     * parent2: reference to the second parent
     */
    fn crossover(parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let mut child1 = Individual::new();
        let mut child2 = Individual::new();
    
        for i in 0..KS_CAPACITY {
            if i < CROSSOVER_POINT {
                child1.genes[i] = parent1.genes[i];
                child2.genes[i] = parent2.genes[i];
            } else {
                child1.genes[i] = parent2.genes[i];
                child2.genes[i] = parent1.genes[i];
            }
        }
    
        (child1, child2)
    }
    
    /**
     * Pick the best parents based on fitness using the roulette wheel selection method
     * Returns the indices of the two parents
     */
  /*  fn pick_best_parents_roulette(&self) -> (usize, usize) {
        let mut rng = rand::thread_rng();

        // Calculate total fitness of the population
        let total_fitness: f64 = self.individuals.iter().map(|ind| ind.fitness).sum();

        // If total fitness is 0, fallback to random selection
        if total_fitness == 0.0 {
            let parent1 = rng.gen_range(0..self.size);
            let mut parent2 = rng.gen_range(0..self.size);
            while parent1 == parent2 {
                parent2 = rng.gen_range(0..self.size);
            }
            return (parent1, parent2);
        }

        // Private helper function to pick one parent based on fitness
        fn pick_one(individuals: &[Individual], total_fitness: f64, rng: &mut impl Rng) -> usize {
            let mut target = rng.gen_range(0.0..total_fitness);
            for (index, individual) in individuals.iter().enumerate() {
                target -= individual.fitness;
                if target <= 0.0 {
                    return index;
                }
            }
            individuals.len() - 1 // Fallback (shouldn't happen if fitness values are consistent)
        }

        // Pick two distinct parents
        let parent1 = pick_one(&self.individuals, total_fitness, &mut rng);
        let mut parent2 = pick_one(&self.individuals, total_fitness, &mut rng);
        while parent1 == parent2 {
            parent2 = pick_one(&self.individuals, total_fitness, &mut rng);
        }

        (parent1, parent2)
    }
*/

    /**
     * Pick the best parents based on fitness using the rank selection method
     * The rank selection method assigns a rank to each individual based on their fitness
     * and then selects parents based on the rank after normalizing the ranks
     * Returns the indices of the two parents
     */
    fn pick_best_parents_rank(&self) -> (usize, usize) {
        let mut rng = rand::thread_rng();
    
        // Sort individuals by descending fitness: best at sorted_indices[0]
        let mut sorted_indices: Vec<_> = (0..self.size).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.individuals[b]
                .fitness
                .partial_cmp(&self.individuals[a].fitness)
                .unwrap()
        });
    
        // Ranks in descending order: best gets rank = size, worst gets rank = 1
        let ranks: Vec<f64> = (1..=self.size)
            .rev()
            .map(|i| i as f64)
            .collect(); 
    
        // Normalizes ranks to probabilities
        let total_rank: f64 = ranks.iter().sum(); 
        let probabilities: Vec<f64> = ranks.iter().map(|r| r / total_rank).collect();
    
        /**
         * Private helper function to pick one parent based on rank
         * Returns the index of the selected parent
         */
        fn pick_one(
            sorted_indices: &[usize],
            probabilities: &[f64],
            rng: &mut impl Rng,
        ) -> usize {
            let target = rng.gen::<f64>();
            let mut cumulative_probability = 0.0;
            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumulative_probability += probabilities[i];
                if target <= cumulative_probability {
                    return idx;
                }
            }
            *sorted_indices.last().unwrap() // Fallback
        }
    
        // Picks two distinct parents based on rank
        let parent1 = pick_one(&sorted_indices, &probabilities, &mut rng);
        let mut parent2 = pick_one(&sorted_indices, &probabilities, &mut rng);

        // Ensure parents are distinct by picking again if necessary (in case of the same parent being picked twice)  
        while parent1 == parent2 {
            parent2 = pick_one(&sorted_indices, &probabilities, &mut rng);
        }
    
        (parent1, parent2)
    }
    
    /**
     * Perform crossover on the population to create new individuals
     * Returns a vector of new individuals
     * The crossover method used here is single-point crossover
     * where the crossover point is at the middle of the genes - based on the
     * constant CROSSOVER_POINT defined above
     */
    fn perform_crossover(&self) -> Vec<Individual> {
        let mut new_individuals = Vec::new();
    
        for _ in 0..self.individuals.len() / 2 {
            let (parent1, parent2) = self.pick_best_parents_rank();
            let (child1, child2) = Self::crossover(&self.individuals[parent1], &self.individuals[parent2]);
        
            new_individuals.push(child1);
            new_individuals.push(child2);
        }
        
        new_individuals
    }

    /**
     * Perform mutation on the new individuals
     * Returns a vector of mutated individuals
     */
    fn perform_mutation(&mut self, mut new_individuals: Vec<Individual>) -> Vec<Individual> {
        let mut rng = rand::thread_rng();
    
        for individual in &mut new_individuals {
            for i in 0..KS_CAPACITY {
                if rng.gen::<f64>() < MUTATION_RATE {
                    individual.genes[i] = 1 - individual.genes[i];
                }
            }
        }
    
        new_individuals
    }

    /**
     * Calculate the diversity of the population using entropy
     * Returns the entropy value as a measure of diversity
     */
    pub fn get_diversity_entropy(&self) -> f64 {
        let mut gene_counts = [0; KS_CAPACITY];
    
        // Count the number of '1' bits at each gene position
        for individual in &self.individuals {
            for i in 0..KS_CAPACITY {
                gene_counts[i] += individual.genes[i] as i32;
            }
        }
    
        // Calculate the entropy based on the gene counts
        let mut entropy = 0.0;
        for &count_ones in gene_counts.iter() {
            // p = fraction of '1's at this gene position
            let p = count_ones as f64 / self.size as f64;
            // fraction of '0's
            let q = 1.0 - p;
    
            // Only adds terms if they are > 0, to avoid NaNs from log2(0)
            if p > 0.0 {
                entropy -= p * p.log2();
            }
            if q > 0.0 {
                entropy -= q * q.log2();
            }
        }
        entropy
    }

    /**
     * Get the fitness statistics of the population
     * Returns a tuple of the minimum, mean, and maximum fitness values
     */
    pub fn get_fitness_stats(&self) -> (f64, f64, f64) {
        // If population can be empty, handle that case
        let min_fitness = self.individuals.iter().map(|ind| ind.fitness).fold(f64::MAX, f64::min);
        let max_fitness = self.individuals.iter().map(|ind| ind.fitness).fold(f64::MIN, f64::max);
        let sum_fitness: f64 = self.individuals.iter().map(|ind| ind.fitness).sum();
        let mean_fitness = sum_fitness / self.individuals.len() as f64;

        (min_fitness, mean_fitness, max_fitness)
    }

    /**
     * Public function to evolve the population
     * items: vector of items in the knapsack
     * This function performs the following steps:
     * 1. Evaluate the fitness of the population
     * 2. Sort the individuals based on fitness
     * 3. Keep the top 10% of the population as elites
     * 4. Perform crossover on the remaining 90% of the population
     * 5. Perform mutation on the new offspring
     * 6. Combine the elites with the new offspring
     * Elitism is used to ensure that the best individuals are preserved in the next generation
     */
    pub fn evolve(&mut self, items: &[Item]) {
        // Evaluate first, so we know who are the best
        self.evaluate_fitness(items);
        self.individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    
        // Keep top 10%
        let elite_count = self.size / 10; 
        let elites: Vec<_> = self.individuals[..elite_count].to_vec();
    
        // Create new offspring
        let new_individuals = self.perform_crossover();
        let mutated_individuals = self.perform_mutation(new_individuals);
    
        // Combine elites with new offspring
        self.individuals = elites.into_iter().chain(mutated_individuals).take(self.size).collect();
    
        // Evaluate the new population
        self.evaluate_fitness(items);
    }
    
    /**
     * Get the best individual in the population
     * Returns a reference to the best individual
     */
    pub fn get_best_individual(&self) -> Option<&Individual> {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }
    
}
