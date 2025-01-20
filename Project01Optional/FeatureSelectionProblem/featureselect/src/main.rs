
mod params;
mod individual;

use std::error::Error;
use plotters::prelude::*;
use std::collections::HashMap;

mod utils;
use utils::{read_txt, compute_rmse_all, compute_rmse_subset};
use params::POPULATION_SIZE;

mod population;
use population::Population; // The GA approach for feature selection

fn main() -> Result<(), Box<dyn Error>> {
    //---------------------------------------------
    // 1) Read the dataset from .txt
    //---------------------------------------------
    let dataset = read_txt(
        "/home/oo/Bio.AI/project-1/project_1/Project01Optional/FeatureSelectionProblem/featureselect/src/feature_selection/dataset.txt", 
        102  // 101 features + 1 target
    )?;
    let n = dataset.len();
    if n == 0 {
        return Err("Dataset is empty".into());
    }
    let m = dataset[0].features.len(); // should be 101

    // Flatten features + targets
    let mut feature_data = Vec::with_capacity(n * m);
    let mut target_data = Vec::with_capacity(n);
    for rec in dataset {
        feature_data.extend_from_slice(&rec.features);
        target_data.push(rec.target);
    }

    //---------------------------------------------
    // 2) Baseline: use all features => compute RMSE
    //---------------------------------------------
    let rmse_all = compute_rmse_all(&feature_data, &target_data, n, m)?;
    println!("Baseline (all columns) RMSE: {:.4}", rmse_all);

    //---------------------------------------------
    // 3) Create a GA Population for feature selection
    //---------------------------------------------
    let mut population = Population::new(POPULATION_SIZE);

    let mut cache: HashMap<String, f64> = HashMap::new();

    // For logging progress
    let mut min_fitness_history = Vec::new();
    let mut mean_fitness_history = Vec::new();
    let mut max_fitness_history = Vec::new();
    let mut diversity_history = Vec::new();

    // Optional: track improvement to do early-stopping
    let mut best_fitness_so_far = -999999.0; 
    let mut stagnant_generations = 0;

    //---------------------------------------------
    // 4) GA Loop: run up to 30 generations
    //---------------------------------------------
    for generation in 0..100 {
        // Evaluate population -> each individual's fitness
        // (calls each individual's `calculate_fitness`, 
        //  which uses negative RMSE as the "fitness")
        population
            .evaluate_fitness(&feature_data, &target_data, n, m, &mut cache)
            ?;

        // Check best individual
        if let Some(best_ind) = population.get_best_individual() {
            // a *higher* "fitness" => a *lower* RMSE.
            let current_best_fitness = best_ind.fitness;
            let current_rmse = -current_best_fitness; 


            if generation % 2 == 0 {
                println!(
                    "Generation {}: Best RSME = {:.4}, Diversity = {:.2}, Size = {}",
                    generation,
                    current_rmse,
                    population.get_diversity_entropy(),
                    population.size
                );
                // let mut active_cols = Vec::new();
                // for (col_idx, &bit) in best_ind.genes.iter().enumerate() {
                //     if bit == 1 {
                //         active_cols.push(col_idx);
                //     }
                // }
                // // Now compute the *full dataset* RMSE with those columns
                // let current_best_rmse = compute_rmse_subset(
                //     &feature_data,     // the entire data
                //     &target_data,
                //     n,                 // total rows
                //     m,                 // total columns
                //     &active_cols,
                // )?;
                // println!("Best individual RMSE (entire data): {:.4}", current_best_rmse);
            }

            

            // If fitness improved by at least 2%, reset stagnation
            let improvement_threshold = best_fitness_so_far * 1.02;
            if current_best_fitness >= improvement_threshold {
                best_fitness_so_far = current_best_fitness;
                stagnant_generations = 0;
            } else {
                stagnant_generations += 1;
            }
        }

        // Log stats for plotting
        let (min_f, mean_f, max_f) = population.get_fitness_stats();
        min_fitness_history.push(min_f);
        mean_fitness_history.push(mean_f);
        max_fitness_history.push(max_f);
        diversity_history.push(population.get_diversity_entropy());

        // GA step: evolve
        population.evolve(&feature_data, &target_data, n, m, &mut cache)?;

        // Early stopping if no improvement for X gens
        if stagnant_generations >= 50 {
            println!("No >=2% improvement for 50 consecutive gens. Stopping at gen {}", generation);
            break;
        }
    }

    //---------------------------------------------
    // 5) Plot GA fitness stats ( min/mean/max ) 
    //---------------------------------------------
    plot_fitness_stats(
        &min_fitness_history,
        &mean_fitness_history,
        &max_fitness_history,
        "ga_feature_fitness.png",
        "GA Fitness (Feature Selection)"
    )?;

    //---------------------------------------------
    // 6) Plot Diversity (entropy)
    //---------------------------------------------
    plot_diversity_history(
        &diversity_history,
        "ga_feature_diversity.png",
        "GA Diversity Over Gens"
    )?;

    Ok(())
}

//----------------------------------
// Plotting Helpers
//----------------------------------

fn plot_fitness_stats(
    min_f: &[f64],
    mean_f: &[f64],
    max_f: &[f64],
    filename: &str,
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    // Determine the overall min and max fitness values to scale the y-axis properly
    let overall_min = min_f
        .iter()
        .chain(mean_f.iter())
        .chain(max_f.iter())
        .copied()
        .fold(f64::MAX, f64::min);
    let overall_max = min_f
        .iter()
        .chain(mean_f.iter())
        .chain(max_f.iter())
        .copied()
        .fold(f64::MIN, f64::max);

    // Add a small margin for better visualization
    let margin = (overall_max - overall_min) * 0.1;
    let y_min = overall_min - margin;
    let y_max = overall_max + margin;

    let root_area = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption(caption, ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..(max_f.len() as i32), y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Draw the Min Fitness line
    chart
        .draw_series(LineSeries::new(
            min_f.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &RED,
        ))?
        .label("Min Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // Draw the Mean Fitness line
    chart
        .draw_series(LineSeries::new(
            mean_f.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &GREEN,
        ))?
        .label("Mean Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    // Draw the Max Fitness line
    chart
        .draw_series(LineSeries::new(
            max_f.iter().enumerate().map(|(x, &y)| (x as i32, y)),
            &BLUE,
        ))?
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    println!("GA fitness stats plot saved to '{}'", filename);
    Ok(())
}


fn plot_diversity_history(
    diversity: &[f64],
    filename: &str,
    caption: &str
) -> Result<(), Box<dyn Error>> {
    let max_div = diversity.iter().copied().fold(0.0_f64, f64::max);
    let root_area = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption(caption, ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..(diversity.len() as i32), 0.0..max_div)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        diversity.iter().enumerate().map(|(x, &y)| (x as i32, y)),
        &RED
    ))?
    .label("Diversity")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw()?;

    println!("Diversity plot saved to '{}'", filename);
    Ok(())
}
