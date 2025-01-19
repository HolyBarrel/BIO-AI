mod utils;
mod individual;
mod population;

use std::error::Error;
use crate::utils::read_csv;
use crate::individual::Item;
use crate::population::Population;

use plotters::prelude::*;

const POPULATION_SIZE: usize = 500;

/**
 * Main function to run the genetic algorithm for the knapsack problem
 */
fn main() -> Result<(), Box<dyn Error>> {
    // Reads items from the CSV file
    let items_file = "/home/oo/Bio.AI/project-1/project_1/src/KP/knapPI_12_500_1000_82.csv";
    let items: Vec<Item> = read_csv(items_file)?;

    // Creates a population
    let mut population = Population::new(POPULATION_SIZE);

    // Evaluates fitness of the initial population
    population.evaluate_fitness(&items);


    // Tracks the diversity of the population
    let mut diversity_history = Vec::new();

    // Vectors for  tracking min, mean, max fitness
    let mut min_fitness_history = Vec::new();
    let mut mean_fitness_history = Vec::new();
    let mut max_fitness_history = Vec::new();

    // Track best fitness so far and how many generations since improvement
    let mut best_fitness_so_far = 0.0;
    let mut stagnant_generations = 0;

    // Run up to 201 generations
    for generation in 0..201 {
        // Check the best individual
        if let Some(best_individual) = population.get_best_individual() {
            let current_best = best_individual.fitness;

            // Print best every 20 gens (optional)
            if generation % 20 == 0 {
                println!(
                    "Generation {}: Best Fitness = {:.2}, Diversity = {:.2}, Size = {}",
                    generation,
                    current_best,
                    population.get_diversity_entropy(),
                    population.size
                );
            }

            // Check if we have at least a 5% improvement over the previous best
            let five_percent_threshold = best_fitness_so_far * 1.02;
            if current_best >= five_percent_threshold {
                // We got at least 2% better => reset counters
                best_fitness_so_far = current_best;
                stagnant_generations = 0;
            } else {
                // Otherwise, increment stagnation
                stagnant_generations += 1;
            }
        }

        // Log stats for plotting
        let (min_fit, mean_fit, max_fit) = population.get_fitness_stats();
        min_fitness_history.push(min_fit);
        mean_fitness_history.push(mean_fit);
        max_fitness_history.push(max_fit);

        diversity_history.push(population.get_diversity_entropy());

        // Evolve
        population.evolve(&items);

        // If no >=2% improvement for 60 gens, an early stop is triggered
        if stagnant_generations >= 60 {
            println!(
                "No >=2% improvement for {} consecutive generations. Stopping at generation {}.",
                stagnant_generations,
                generation
            );
            break;
        }
    }


    plot_fitness_stats(&min_fitness_history, &mean_fitness_history, &max_fitness_history)?;
    plot_diversity_history(&diversity_history)?;



    Ok(())
}

/**
 * Function to plot the fitness stats over the generations
 * Saves the plot to a file called 'fitness_stats.png'
 */
fn plot_fitness_stats(
    min_fit: &[f64],
    mean_fit: &[f64],
    max_fit: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("fitness_stats.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Determine the overall max Y for the chart
    let overall_max = max_fit.iter().copied().fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Fitness Score Over Generations (Knapsack Problem)", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0..(max_fit.len() as i32),
            0.0..overall_max,
        )?;

    chart.configure_mesh().draw()?;

    // Draw the min line
    chart
        .draw_series(LineSeries::new(
            min_fit.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &RED,
        ))?
        .label("Min Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // Draw the mean line
    chart
        .draw_series(LineSeries::new(
            mean_fit.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &GREEN,
        ))?
        .label("Mean Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    // Draw the max line
    chart
        .draw_series(LineSeries::new(
            max_fit.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &BLUE,
        ))?
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    println!("Fitness stats plot saved to 'fitness_stats.png'");

    Ok(())
}


/**
 * Function to plot the population diversity history
 * Saves the plot to a file called 'population_diversity.png'
 */
fn plot_diversity_history(diversity_history: &[f64]) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new("population_diversity.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let max_diversity = diversity_history
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Population Diversity Over Generations", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0..(diversity_history.len() as i32),
            0.0..max_diversity
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        diversity_history
            .iter()
            .enumerate()
            .map(|(x, &y)| (x as i32, y)),
        &RED,
    ))?
    .label("Population Diversity")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    println!("Population diversity plot saved to 'population_diversity.png'");

    Ok(())
}
