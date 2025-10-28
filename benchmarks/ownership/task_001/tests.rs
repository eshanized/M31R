#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet(String::from("hello")), "hello world");
    }

    #[test]
    fn test_empty() {
        assert_eq!(greet(String::from("")), " world");
    }
}
