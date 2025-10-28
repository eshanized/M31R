// This function has a bug. Fix it so it returns the maximum
// value in the slice. It currently panics on empty slices.
fn find_max(values: &[i32]) -> Option<i32> {
    if values.is_empty() {
        return None;
    }
    let mut max = values[0];
    // Bug: starts at 0 instead of 1, comparing element with itself
    for i in 0..values.len() {
        if values[i] > max {
            max = values[i];
        }
    }
    Some(max)
}

fn main() {}
