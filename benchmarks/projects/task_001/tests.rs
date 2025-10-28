#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let c = Counter::new();
        assert_eq!(c.value(), 0);
    }

    #[test]
    fn test_increment() {
        let mut c = Counter::new();
        c.increment();
        c.increment();
        assert_eq!(c.value(), 2);
    }
}
