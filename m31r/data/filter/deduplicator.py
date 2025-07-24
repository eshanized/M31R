# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Hash-based content deduplication.

The goal here is simple: if two files have identical content, keep only one.
This catches forks, mirrors, vendored copies, and copy-pasted code across
repos. Without dedup, the model would overfit on commonly copied snippets.

We use SHA256 hashes of the file content to detect duplicates. This means
we only store a set of 64-character hex strings in memory — not the file
contents themselves. Even with millions of files, the hash set stays small
enough to fit in RAM comfortably.
"""

from m31r.utils.hashing import compute_sha256_bytes


class ContentDeduplicator:
    """
    Tracks file content hashes to detect and reject duplicates.

    Usage is straightforward: call is_duplicate() with each file's content.
    First time it sees a particular content, it returns False and remembers
    the hash. Second time (or third, or hundredth), it returns True.

    The deduplicator is stateful — it accumulates hashes across calls.
    Create a fresh instance for each pipeline run to start with a clean slate.
    """

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def is_duplicate(self, content: str) -> bool:
        """
        Check if we've seen this exact content before.

        The content gets hashed with SHA256, so we're comparing 64-byte
        hex digests rather than full file contents. Two files with even
        a single byte difference will have completely different hashes.
        """
        content_hash = compute_sha256_bytes(content.encode("utf-8"))
        if content_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(content_hash)
        return False

    @property
    def unique_count(self) -> int:
        """How many unique files have we seen so far."""
        return len(self._seen_hashes)

    def reset(self) -> None:
        """Clear all tracked hashes and start fresh."""
        self._seen_hashes.clear()
