use std::error::Error;
use csv::{ReaderBuilder, StringRecord};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::mean_squared_error;

#[derive(Debug)]
pub struct Record {
    pub features: Vec<f64>,
    pub target: f64,
}

/// read_txt: reads a 102-col comma-delimited file,
/// the first 101 as features, last as target
pub fn read_txt(path: &str, total_cols: usize) -> Result<Vec<Record>, Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    // use default comma-delimiter, no headers
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut records = Vec::new();

    for row in rdr.records() {
        let record: StringRecord = row?;
        if record.len() != total_cols {
            return Err(format!(
                "Expected {} columns, got {} in line {:?}",
                total_cols, record.len(), record
            )
            .into());
        }
        let target_index = total_cols - 1;
        let target_val = record[target_index].parse::<f64>()?;
        let mut feats = Vec::with_capacity(target_index);
        for i in 0..target_index {
            feats.push(record[i].parse::<f64>()?);
        }
        records.push(Record {
            features: feats,
            target: target_val,
        });
    }
    Ok(records)
}

/// compute_rmse_all builds a DenseMatrix from a flattened feature_data
/// plus a target_data vec. Then it fits linear regression on *all* columns,
/// returning the RMSE.
pub fn compute_rmse_all(
    feature_data: &[f64],
    target_data: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<f64, Box<dyn Error>> {
    // Build a DenseMatrix with owned f64
    // row_major = false => last param is false
    // if you want row-major, pass false
    let mat = DenseMatrix::new(nrows, ncols, feature_data.to_vec(), false)
        .map_err(|e| format!("Failed to build matrix: {:?}", e))?;

    // Fit linear regression
    let linreg = LinearRegression::fit(&mat, &target_data.to_vec(), Default::default())
        .map_err(|e| format!("Fit failed: {:?}", e))?;

    // Predict
    let preds = linreg
        .predict(&mat)
        .map_err(|e| format!("Predict failed: {:?}", e))?;

    // MSE => RMSE
    let mse = mean_squared_error(&target_data.to_vec(), &preds);
    let rmse = mse.sqrt();
    Ok(rmse)
}

/// compute_rmse_subset builds a smaller DenseMatrix from a subset of columns.
/// Then it trains linear regression on that submatrix + the provided target_data,
/// returning the RMSE.
///
/// * `feature_data`: flattened row-major data of size nrows * ncols
/// * `target_data`: a Vec<f64> of length nrows
/// * `nrows`: total rows
/// * `ncols`: total columns in the original data
/// * `active_cols`: the columns we want to keep in the submatrix
pub fn compute_rmse_subset(
    feature_data: &[f64],
    target_data: &[f64],
    nrows: usize,
    ncols: usize,
    active_cols: &[usize],
) -> Result<f64, Box<dyn std::error::Error>> {

    // 1) If no columns selected, return error
    if active_cols.is_empty() {
        return Err("No columns selected".into());
    }

    // 2) Build sub_data for just those columns
    //    For each row in 0..nrows,
    //    we pick out the columns in `active_cols`.
    let mut sub_data = Vec::with_capacity(nrows * active_cols.len());
    for row in 0..nrows {
        for &col in active_cols {
            let idx = row * ncols + col;
            sub_data.push(feature_data[idx]);
        }
    }

    // 3) Construct the submatrix
    // row_major = false => last param is false
    let mat = DenseMatrix::new(nrows, active_cols.len(), sub_data, false)
        .map_err(|e| format!("Failed to build submatrix: {:?}", e))?;

    // 4) Fit linear regression on the submatrix
    let linreg = LinearRegression::fit(&mat, &target_data.to_vec(), Default::default())
        .map_err(|e| format!("Fit failed: {:?}", e))?;

    // 5) Predict and compute RMSE
    let preds = linreg
        .predict(&mat)
        .map_err(|e| format!("Predict failed: {:?}", e))?;
    let mse = mean_squared_error(&target_data.to_vec(), &preds);
    let rmse = mse.sqrt();

    Ok(rmse)
}

pub fn compute_rmse_subset_direct(
    feature_data: &[f64],   // row-major submatrix data
    target_data: &[f64],    // partial target
    nrows: usize,
    ncols: usize,
) -> Result<f64, Box<dyn Error>> {

    // 1) Build a DenseMatrix from partial data
    let mat = DenseMatrix::new(nrows, ncols, feature_data.to_vec(), false)
        .map_err(|e| format!("Failed to build partial matrix: {:?}", e))?;

    // 2) Fit LR
    let linreg = LinearRegression::fit(&mat, &target_data.to_vec(), Default::default())
        .map_err(|e| format!("LinearRegression fit failed: {:?}", e))?;

    // 3) Predict
    let preds = linreg
        .predict(&mat)
        .map_err(|e| format!("Predict failed: {:?}", e))?;
    
    // 4) MSE => RMSE
    let mse = mean_squared_error(&target_data.to_vec(), &preds);
    let rmse = mse.sqrt();
    Ok(rmse)
}
