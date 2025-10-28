#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn test_five() {
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_one() {
        assert_eq!(factorial(1), 1);
    }
}
