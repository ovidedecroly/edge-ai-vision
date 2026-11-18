from collections import defaultdict
from typing import List, Dict, Any, Tuple

class TemporalVoter:
    """
    Multi-frame Temporal Voting across tracked vehicle frames.
    Weights OCR character hypotheses by character confidence and quality score.
    Eliminates OCR single-frame jitter.
    """
    def __init__(self, min_votes: int = 2):
        self.min_votes = min_votes
        self.candidates_by_track: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_candidate(self, track_id: str, text: str, confidence: float, quality_score: float = 1.0):
        if not text:
            return
        self.candidates_by_track[track_id].append({
            "text": text,
            "confidence": confidence,
            "quality_score": quality_score,
            "weight": confidence * quality_score
        })

    def get_consensus(self, track_id: str) -> Tuple[str, float, int]:
        """
        Returns (consensus_text, consensus_confidence, vote_count).
        """
        candidates = self.candidates_by_track.get(track_id, [])
        if not candidates:
            return "", 0.0, 0

        # Frequency and weight dictionary
        weighted_scores = defaultdict(float)
        vote_counts = defaultdict(int)

        for c in candidates:
            text = c["text"]
            weighted_scores[text] += c["weight"]
            vote_counts[text] += 1

        best_text = max(weighted_scores, key=weighted_scores.get)
        total_weight = weighted_scores[best_text]
        best_count = vote_counts[best_text]
        avg_confidence = min(0.99, total_weight / best_count)

        return best_text, round(float(avg_confidence), 3), best_count

    def clear_track(self, track_id: str):
        if track_id in self.candidates_by_track:
            del self.candidates_by_track[track_id]
