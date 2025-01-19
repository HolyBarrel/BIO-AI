/// The maximum capacity of the knapsack
pub const WEIGHT_CAPACITY: i32 = 280785;

/// The total number of genes/items in the knapsack
pub const KS_CAPACITY: usize = 500;

/// The crossover point is set to half of KS_CAPACITY
/// to ensure that the offspring inherits 50% first of the genes from
/// parent 1 and the second 50% of the genes belonging to the second parent
pub const CROSSOVER_POINT: usize = KS_CAPACITY / 2;

/// The mutation rate for offspring
pub const MUTATION_RATE: f64 = 1.0 / KS_CAPACITY as f64;

/// The population size for the generations in the SGA
pub const POPULATION_SIZE: usize = 500;