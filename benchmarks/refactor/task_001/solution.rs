fn sum_even(numbers: &[i32]) -> i32 {
    numbers.iter().filter(|&&x| x % 2 == 0).sum()
}

fn main() {}
