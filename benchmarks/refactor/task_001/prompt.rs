// Refactor this imperative code to use iterators instead of a manual loop.
fn sum_even(numbers: &[i32]) -> i32 {
    let mut total = 0;
    for i in 0..numbers.len() {
        if numbers[i] % 2 == 0 {
            total += numbers[i];
        }
    }
    total
}

fn main() {}
