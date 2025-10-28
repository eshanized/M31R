#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed() {
        assert_eq!(sum_even(&[1, 2, 3, 4, 5, 6]), 12);
    }

    #[test]
    fn test_empty() {
        assert_eq!(sum_even(&[]), 0);
    }

    #[test]
    fn test_all_odd() {
        assert_eq!(sum_even(&[1, 3, 5]), 0);
    }
}
