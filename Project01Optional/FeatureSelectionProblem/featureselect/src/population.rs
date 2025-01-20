use rand::Rng;
use crate::individual::Individual;
use crate::params::{NUM_FEATURES, CROSSOVER_POINT, MUTATION_RATE};
use std::collections::HashMap;
use crate::Error;

#[derive(Debug, Clone)]
pub struct Population {
    pub individuals: Vec<Individual>,
    pub size: usize,
}

impl Population {
    /// Constructor for Population: generate 'size' random Individuals
    pub fn new(size: usize) -> Self {
        let individuals = (0..size)
            .map(|_| Individual::new())
            .collect();
        Population { individuals, size }
    }

    /// Evaluate the fitness of each Individual by calling `calculate_fitness`
    /// - feature_data, target_data: flattened features + targets
    /// - nrows, ncols: shape of the full dataset
    pub fn evaluate_fitness(
        &mut self,
        feature_data: &[f64],
        target_data: &[f64],
        nrows: usize,
        ncols: usize,
        cache: &mut HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for ind in &mut self.individuals {
            ind.calculate_fitness(feature_data, target_data, nrows, ncols, cache)?;
        }
        Ok(())
    }

    /// Single-point crossover
    fn crossover(parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let mut child1 = Individual::new();
        let mut child2 = Individual::new();

        for i in 0..NUM_FEATURES {
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

    /// Pick two parents via rank selection
    fn pick_best_parents_rank(&self) -> (usize, usize) {
        let mut rng = rand::thread_rng();
        // Sort by descending fitness
        let mut sorted_indices: Vec<_> = (0..self.size).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.individuals[b]
                .fitness
                .partial_cmp(&self.individuals[a].fitness)
                .unwrap()
        });
        // Create descending ranks, then convert to probabilities
        let ranks: Vec<f64> = (1..=self.size).rev().map(|x| x as f64).collect();
        let total_rank: f64 = ranks.iter().sum();
        let probabilities: Vec<f64> = ranks.iter().map(|r| r / total_rank).collect();

        fn pick_one(
            sorted_indices: &[usize],
            probabilities: &[f64],
            rng: &mut impl Rng,
        ) -> usize {
            let target = rng.gen::<f64>();
            let mut cumulative = 0.0;
            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumulative += probabilities[i];
                if target <= cumulative {
                    return idx;
                }
            }
            *sorted_indices.last().unwrap()
        }

        let parent1 = pick_one(&sorted_indices, &probabilities, &mut rng);
        let mut parent2 = pick_one(&sorted_indices, &probabilities, &mut rng);
        while parent1 == parent2 {
            parent2 = pick_one(&sorted_indices, &probabilities, &mut rng);
        }
        (parent1, parent2)
    }

    /// Measure genetic diversity (entropy) across each gene
    pub fn get_diversity_entropy(&self) -> f64 {
        let mut gene_counts = [0; NUM_FEATURES];
        for ind in &self.individuals {
            for i in 0..NUM_FEATURES {
                gene_counts[i] += ind.genes[i] as i32;
            }
        }
        let mut entropy = 0.0;
        for &count in &gene_counts {
            let p = count as f64 / self.size as f64;
            let q = 1.0 - p;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
            if q > 0.0 {
                entropy -= q * q.log2();
            }
        }
        entropy
    }

    /// Get (min, mean, max) fitness
    pub fn get_fitness_stats(&self) -> (f64, f64, f64) {
        let min_f = self.individuals.iter().map(|i| i.fitness).fold(f64::MAX, f64::min);
        let max_f = self.individuals.iter().map(|i| i.fitness).fold(f64::MIN, f64::max);
        let sum_f: f64 = self.individuals.iter().map(|i| i.fitness).sum();
        let mean_f = sum_f / (self.individuals.len() as f64);
        (min_f, mean_f, max_f)
    }

    /// Computes how many bits differ between two bit-strings.
    fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
        a.iter()
            .zip(b.iter())
            .filter(|(&x, &y)| x != y)
            .count()
    }

    fn crowding_replacement(
        parent1: &Individual,
        parent2: &Individual,
        child1: &mut Individual,
        child2: &mut Individual,
        feature_data: &[f64],
        target_data: &[f64],
        nrows: usize,
        ncols: usize,
        cache: &mut HashMap<String, f64>,
    ) -> Result<(Individual, Individual), Box<dyn Error>> {

        //--- Evaluate the children's fitness so they're up to date
        //    (the parents' fitness should already be known)
        child1.calculate_fitness(feature_data, target_data, nrows, ncols, cache)?;
        child2.calculate_fitness(feature_data, target_data, nrows, ncols, cache)?;

        //--- Measure distances
        let d_p1_c1 = Self::hamming_distance(&parent1.genes, &child1.genes);
        let d_p1_c2 = Self::hamming_distance(&parent1.genes, &child2.genes);
        let d_p2_c1 = Self::hamming_distance(&parent2.genes, &child1.genes);
        let d_p2_c2 = Self::hamming_distance(&parent2.genes, &child2.genes);

        //--- Decide pairing
        //
        // If parent1 is closer to child1 *and* parent2 is closer to child2,
        //   pair (p1,c1) and (p2,c2).
        // Otherwise, pair (p1,c2) and (p2,c1).
        let (p1_candidate, c1_candidate, p2_candidate, c2_candidate) =
            if d_p1_c1 + d_p2_c2 <= d_p1_c2 + d_p2_c1 {
                (parent1, child1, parent2, child2)
            } else {
                (parent1, child2, parent2, child1)
            };

        //--- For each pair, pick the fitter. Remember you define “higher fitness” as better.
        let survivor1 = if p1_candidate.fitness >= c1_candidate.fitness {
            p1_candidate.clone()
        } else {
            c1_candidate.clone()
        };
        let survivor2 = if p2_candidate.fitness >= c2_candidate.fitness {
            p2_candidate.clone()
        } else {
            c2_candidate.clone()
        };

        Ok((survivor1, survivor2))
    }


    fn mutate_individual(ind: &mut Individual) {
        let mut rng = rand::thread_rng();
        for i in 0..NUM_FEATURES {
            if rng.gen::<f64>() < MUTATION_RATE {
                ind.genes[i] = 1 - ind.genes[i];
            }
        }
    }
    

    /// Evolve the population:
    /// 1) Evaluate
    /// 2) Sort by fitness, keep top 10% as elites
    /// 3) Crossover + mutation
    /// 4) Replace population
    /// 5) Evaluate again
    pub fn evolve(
        &mut self,
        feature_data: &[f64],
        target_data: &[f64],
        nrows: usize,
        ncols: usize,
        cache: &mut HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1) Evaluate the current population's fitness
        self.evaluate_fitness(feature_data, target_data, nrows, ncols, cache)?;
        
        // 2) Sort descending by fitness and keep top 10% as elites
        self.individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let elite_count = self.size / 10;
        let elites = self.individuals[..elite_count].to_vec();
    
        // 3) We'll build new population with these elites plus
        //    pairs that survive crowding. 
        let mut new_population = Vec::with_capacity(self.size);
        new_population.extend(elites);
    
        // 4) We do pairwise selection & crowding for the other 90%.
        //    We'll produce (size - elite_count) individuals in pairs.
        while new_population.len() < self.size {
            // a) pick two parents
            let (p1_idx, p2_idx) = self.pick_best_parents_rank();
            let parent1 = self.individuals[p1_idx].clone();
            let parent2 = self.individuals[p2_idx].clone();
            
            // b) produce two children by crossover
            let (mut child1, mut child2) = Self::crossover(&parent1, &parent2);
            
            // c) mutate those children
            Self::mutate_individual(&mut child1);
            Self::mutate_individual(&mut child2);
    
            // d) run crowding replacement => returns 2 survivors
            let (winner1, winner2) = Self::crowding_replacement(
                &parent1,
                &parent2,
                &mut child1,
                &mut child2,
                feature_data,
                target_data,
                nrows,
                ncols,
                cache
            )?;
    
            // e) push them into new population (limit new_population to self.size)
            if new_population.len() < self.size {
                new_population.push(winner1);
            }
            if new_population.len() < self.size {
                new_population.push(winner2);
            }
        }
    
        // 5) Replace old population
        self.individuals = new_population;
    
        // 6) Optionally re-evaluate or rely on the “crowding_replacement” fitness calls
        //    to keep fitness in sync. Often, you might skip a second evaluation pass.
        Ok(())
    }
    


    /// Return the best individual
    pub fn get_best_individual(&self) -> Option<&Individual> {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }
}
