#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal() {
        assert_eq!(find_max(&[1, 3, 2]), Some(3));
    }

    #[test]
    fn test_empty() {
        assert_eq!(find_max(&[]), None);
    }

    #[test]
    fn test_single() {
        assert_eq!(find_max(&[42]), Some(42));
    }
}
