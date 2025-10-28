#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse() {
        assert_eq!(reverse_string("hello"), "olleh");
    }

    #[test]
    fn test_empty() {
        assert_eq!(reverse_string(""), "");
    }
}
