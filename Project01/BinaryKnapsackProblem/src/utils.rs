use std::error::Error;
use std::fs::File;
use std::path::Path;
use crate::individual::Item;

/**
 * Util method to read items from a CSV file
 */
pub fn read_csv<P: AsRef<Path>>(filename: P) -> Result<Vec<Item>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut items = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let item = Item {
            value: record[1].parse()?,
            weight: record[2].parse()?,
        };
        items.push(item);
    }

    Ok(items)
}
