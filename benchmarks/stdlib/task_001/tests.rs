#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort() {
        assert_eq!(sort_ascending(vec![3, 1, 2]), vec![1, 2, 3]);
    }

    #[test]
    fn test_empty() {
        assert_eq!(sort_ascending(vec![]), vec![]);
    }

    #[test]
    fn test_already_sorted() {
        assert_eq!(sort_ascending(vec![1, 2, 3]), vec![1, 2, 3]);
    }
}
