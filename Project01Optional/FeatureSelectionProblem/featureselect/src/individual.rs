use crate::params::{NUM_FEATURES};
use crate::utils::compute_rmse_subset;
use crate::Error;
use std::collections::HashMap;
use rand::{Rng, seq::SliceRandom};
use rand::rngs::StdRng;

/**
 * Individual struct for each of the knapsack problem population members
 * genes: array of genes representing the items in the knapsack
 * fitness: fitness value of the individual, used to evaluate the quality of the solution
 */
#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: [u8; 101], 
    pub fitness: f64,  // is the negative of the rmse, meaning that the value should be optimized
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
    fn generate_random_genes() -> [u8; NUM_FEATURES] {
        let mut genes = [0; NUM_FEATURES];
        let mut rng = rand::thread_rng();

        for i in 0..NUM_FEATURES {
            genes[i] = rng.gen_range(0..=1);
        }

        genes
    }

    fn bitstring_to_string(genes: &[u8]) -> String {
        let mut s = String::with_capacity(genes.len());
        for &g in genes {
            if g == 1 {
                s.push('1');
            } else {
                s.push('0');
            }
        }
        s
    }
    

   /// Evaluate fitness with partial row sampling (~10%) 
    /// plus columns from the bitstring.
    pub fn calculate_fitness(
        &mut self,
        feature_data: &[f64],
        target_data: &[f64],
        nrows: usize,
        ncols: usize,
        cache: &mut HashMap<String, f64>,
    ) -> Result<f64, Box<dyn Error>> {

        // 1) Convert bitstring to a string key (for caching, if you want)
        let subset_key = Self::bitstring_to_string(&self.genes);

        // 2) Check cache (optional)
        if let Some(&cached_fit) = cache.get(&subset_key) {
            // Already computed. Just set self.fitness to that
            self.fitness = cached_fit;
            return Ok(-cached_fit); // Or however you handle negative vs positive
        }

        // 3) Collect active columns from the bitstring
        let mut active_cols = Vec::new();
        for (col_idx, &bit) in self.genes.iter().enumerate() {
            if bit == 1 {
                active_cols.push(col_idx);
            }
        }
        if active_cols.is_empty() {
            // If no columns, fitness is useless
            self.fitness = -9999.0; // big negative or handle differently
            return Ok(9999.0);
        }
        // 4) Fit a linear regression on this submatrix
        let rmse = compute_rmse_subset(
            &feature_data,
            &target_data,
            nrows,
            ncols,
            &active_cols
        )?;

        self.fitness = -rmse;


        // 5) Save negative rmse => so we maximize fitness

        // 6) Cache it if you want
        cache.insert(subset_key, self.fitness);

        // Return the actual RMSE
        Ok(rmse)
    }

    
    
}
