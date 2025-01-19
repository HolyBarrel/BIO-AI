use rand::Rng;
use crate::params::{KS_CAPACITY, WEIGHT_CAPACITY};

/**
 * Individual struct for each of the knapsack problem population members
 * genes: array of genes representing the items in the knapsack
 * fitness: fitness value of the individual, used to evaluate the quality of the solution
 */
#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: [u8; 500], 
    pub fitness: f64,  
}

/**
 * Item struct for each of the items in the knapsack problem
 * index: index of the item
 * value: value of the item
 * weight: weight of the item
 */
pub struct Item {
    pub value: i32,
    pub weight: i32,
}

/**
 * Implementation of the Individual struct
 */
impl Individual {
    /**
     * Constructor for Individual which generates a random set of genes
     */
    pub fn new() -> Self {
        let genes = Self::generate_random_genes();
        Individual {
            genes,
            fitness: 0.0,
        }
    }

    /**
     * Generate a random set of genes for the individual
     * Private function only used in the constructor
     */
    fn generate_random_genes() -> [u8; KS_CAPACITY] {
        let mut genes = [0; KS_CAPACITY];
        let mut rng = rand::thread_rng();

        for i in 0..KS_CAPACITY {
            genes[i] = rng.gen_range(0..=1);
        }

        genes
    }

    /**
     * Calculate the fitness of the individual based on the items in the knapsack
     * items: reference to the vector of items
     */
    pub fn calculate_fitness(&mut self, items: &[Item]) -> f64 {
        let mut total_value = 0;
        let mut total_weight = 0;

        // Calculate the total value and weight of the items in the knapsack
        for (gene, item) in self.genes.iter().zip(items) {
            if *gene == 1 {
                total_value += item.value;
                total_weight += item.weight;
            }
        }

        // Penalize solutions that exceed the weight capacity
        if total_weight > WEIGHT_CAPACITY {
            // The penalty is the amount by which the weight exceeds the capacity
            let penalty = (total_weight - WEIGHT_CAPACITY) as f64;
            // The fitness is the total value minus the penalty, and 0 if negative
            self.fitness = (total_value as f64 - penalty).max(0.0);
        } else {
            self.fitness = total_value as f64;
        }

        self.fitness
    }
}
