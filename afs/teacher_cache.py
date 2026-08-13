class AFSTeacherCacheBuilder:
    """Policy facade for the upstream in-place GT-assisted Wan KV cache."""

    def __init__(self, shared_prefix_chunks: int = 1):
        if shared_prefix_chunks not in {0, 1}:
            raise ValueError("AFS currently supports shared_prefix_chunks 0 or 1")
        self.shared_prefix_chunks = shared_prefix_chunks

    def replacement_chunk_index(self, target_chunk_index: int):
        """Return the older history chunk to replace, never the current target."""
        candidate = target_chunk_index - 2
        return candidate if candidate >= self.shared_prefix_chunks else None

    def assert_no_target_leakage(self, target_chunk_index: int, cache_chunk_indices) -> None:
        if target_chunk_index in cache_chunk_indices:
            raise RuntimeError(f"Target GT chunk {target_chunk_index} leaked into Teacher visual cache")
