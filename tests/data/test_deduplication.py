# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Deduplication tests.

Verifies that exact-match deduplication works correctly: duplicates get caught,
unique files get kept, and the deduplicator is deterministic.
"""

from m31r.data.filter.deduplicator import ContentDeduplicator


class TestContentDeduplicator:
    def test_first_occurrence_not_duplicate(self) -> None:
        dedup = ContentDeduplicator()
        assert dedup.is_duplicate("fn main() {}") is False

    def test_second_occurrence_is_duplicate(self) -> None:
        dedup = ContentDeduplicator()
        dedup.is_duplicate("fn main() {}")
        assert dedup.is_duplicate("fn main() {}") is True

    def test_different_content_not_duplicate(self) -> None:
        dedup = ContentDeduplicator()
        dedup.is_duplicate("fn main() {}")
        assert dedup.is_duplicate("fn other() {}") is False

    def test_unique_count_tracks_correctly(self) -> None:
        dedup = ContentDeduplicator()
        dedup.is_duplicate("aaa")
        dedup.is_duplicate("bbb")
        dedup.is_duplicate("aaa")
        assert dedup.unique_count == 2

    def test_reset_clears_state(self) -> None:
        dedup = ContentDeduplicator()
        dedup.is_duplicate("content")
        dedup.reset()
        assert dedup.is_duplicate("content") is False
        assert dedup.unique_count == 1

    def test_whitespace_differences_matter(self) -> None:
        dedup = ContentDeduplicator()
        dedup.is_duplicate("fn main() {}")
        assert dedup.is_duplicate("fn main()  {}") is False

    def test_deterministic_hash_order(self) -> None:
        dedup_a = ContentDeduplicator()
        dedup_b = ContentDeduplicator()

        files = ["fn a() {}", "fn b() {}", "fn c() {}"]
        results_a = [dedup_a.is_duplicate(f) for f in files]
        results_b = [dedup_b.is_duplicate(f) for f in files]
        assert results_a == results_b
