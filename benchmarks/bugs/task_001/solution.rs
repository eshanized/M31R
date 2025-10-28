fn find_max(values: &[i32]) -> Option<i32> {
    if values.is_empty() {
        return None;
    }
    let mut max = values[0];
    for i in 1..values.len() {
        if values[i] > max {
            max = values[i];
        }
    }
    Some(max)
}

fn main() {}
