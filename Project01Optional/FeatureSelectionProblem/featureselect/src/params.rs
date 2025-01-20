/// The total number of genes/items in the knapsack
pub const NUM_FEATURES: usize = 101;

/// The crossover point is set to half of NUM_FEATURES
/// to ensure that the offspring inherits 50% first of the genes from
/// parent 1 and the second 50% of the genes belonging to the second parent
pub const CROSSOVER_POINT: usize = (NUM_FEATURES + 1) / 2;

/// The mutation rate for offspring
pub const MUTATION_RATE: f64 = 1.0 / NUM_FEATURES as f64;

/// The population size for the generations in the SGA
pub const POPULATION_SIZE: usize = 12;