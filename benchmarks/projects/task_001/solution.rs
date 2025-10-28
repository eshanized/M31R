struct Counter {
    count: i32,
}

impl Counter {
    fn new() -> Self {
        Counter { count: 0 }
    }

    fn increment(&mut self) {
        self.count += 1;
    }

    fn value(&self) -> i32 {
        self.count
    }
}

fn main() {}
