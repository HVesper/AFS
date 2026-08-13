from .types import AFSTextCondition, AFSTrainingBatch


class AFSTeacherConditionSelector:
    def select(self, batch: AFSTrainingBatch, chunk_index: int, mode: str) -> AFSTextCondition:
        if mode == "global":
            return AFSTextCondition(batch.global_text_embeddings, batch.global_text_masks)
        if mode != "chunk_replace":
            raise ValueError("teacher text condition mode must be global or chunk_replace")
        if chunk_index < 0 or chunk_index >= batch.chunk_text_embeddings.shape[1]:
            raise IndexError(f"Teacher chunk text index {chunk_index} is out of range")
        mask = None if batch.chunk_text_masks is None else batch.chunk_text_masks[:, chunk_index]
        return AFSTextCondition(batch.chunk_text_embeddings[:, chunk_index], mask)
