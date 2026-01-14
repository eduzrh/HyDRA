def calculate_abs_hits_at_1(ref_path: str, pred_path: str) -> float:
    """
    Calculate the Hits@1 metric between two entity alignment files

    Parameters.
        ref_path: reference standard file path
        pred_path: path to the prediction result file

    Returns.
        Hits@1 evaluation value (floating point number)
    """

    def _load_pairs(file_path: str) -> set:
        """Internal function: load set of alignment pairs"""
        pairs = set()
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    pairs.add((parts[0].strip(), parts[1].strip()))
        return pairs

    try:
        ref_pairs = _load_pairs(ref_path)
        pred_pairs = _load_pairs(pred_path)

        correct = len(ref_pairs & pred_pairs)
        total = len(ref_pairs)
        result = correct / total if total > 0 else 0.0
        print(f"Hits@1 = {result:.4f}")
        return correct / total if total > 0 else 0.0

    except FileNotFoundError as e:
        print(f"File Read Error:{e}")
        return 0.0

# Usage example
if __name__ == "__main__":
    ref_path = '/home/dex/Desktop/entity_sy/HyDRA/data/icews_yago/ref_pairs'
    pred_path = '/home/dex/Desktop/entity_sy/HyDRA/data/icews_yago/message_pool/retriever_outputs.txt'

    hits = calculate_abs_hits_at_1(ref_path, pred_path)