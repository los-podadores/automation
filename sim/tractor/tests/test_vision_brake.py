"""Visual stopping policy: actual image results, no recorded-media actuation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_brake import VisionBrake


def packet(sequence=1, stamp=10.0, label=None, **changes):
    return (
        dict(
            mode="vision_brake",
            session_id="session-1",
            source="Gazebo rendered RGB-D",
            frame_sequence=sequence,
            frame_time_s=stamp,
            detections=[dict(label=label, confidence=0.8)] if label else [],
        )
        | changes
    )


class VisionBrakeTests(unittest.TestCase):
    def setUp(self):
        self.guard = VisionBrake(Path("unused"), "session-1")

    def clear(self, sequence, now):
        return self.guard.update(packet(sequence, now), now, now)

    def test_person_and_dog_immediately_stop_even_without_depth(self):
        for index, label in enumerate(("person", "dog")):
            now = 10 + index
            decision = self.guard.update(packet(index + 1, now, label), now, now)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.state, "FRENAR YOLO")
            self.assertIn("persona" if label == "person" else "perro", decision.reason)

    def test_requires_continuous_clear_time_and_multiple_distinct_frames(self):
        self.assertFalse(self.clear(1, 10).allowed)
        self.assertFalse(self.clear(2, 10.7).allowed)
        self.assertTrue(self.clear(3, 11.3).allowed)
        self.guard.update(packet(4, 11.4, "dog"), 11.4, 11.4)
        self.assertFalse(self.clear(5, 11.5).allowed)
        self.assertFalse(self.clear(6, 12.0).allowed)
        self.assertTrue(self.clear(7, 12.8).allowed)

    def test_one_missed_detection_does_not_restart(self):
        self.guard.update(packet(label="person"), 10, 10)
        self.assertEqual(self.clear(2, 10.1).state, "ESPERAR YOLO")
        self.assertEqual(
            self.guard.update(packet(3, 10.2, "person"), 10.2, 10.2).state,
            "FRENAR YOLO",
        )
        self.assertFalse(self.clear(4, 11.0).allowed)

    def test_freezing_or_rewriting_same_frame_never_renews_lease(self):
        self.clear(1, 10)
        payload = packet(written_at=9999999)
        self.assertFalse(self.guard.update(payload, 10, 10.7).allowed)
        self.assertEqual(self.guard.update(payload, 10, 11.1).state, "SIN YOLO")
        self.assertEqual(self.guard.clear_frames, 0)

    def test_missing_replayed_and_modified_results_stop(self):
        self.clear(2, 10)
        for payload in (
            None,
            {},
            packet(1),
            packet(2, 10.1),
            packet(2, label="dog"),
        ):
            with self.subTest(payload=payload):
                self.assertEqual(
                    self.guard.update(payload, 10.1, 10.1).state, "SIN YOLO"
                )

    def test_unrelated_media_or_session_never_authorizes_motion(self):
        for changes in (
            dict(mode="monitor_only"),
            dict(source="PODADORAS.mp4"),
            dict(session_id="other-session"),
        ):
            with self.subTest(changes=changes):
                self.assertFalse(self.guard.update(packet(**changes), 10, 10).allowed)

    def test_capture_age_matters_even_when_file_was_just_received(self):
        for stamp in (8, 11, float("nan"), float("inf"), "10"):
            with self.subTest(stamp=stamp):
                self.assertEqual(
                    self.guard.update(packet(stamp=stamp), 10, 10).state, "SIN YOLO"
                )

    def test_new_sequence_requires_new_capture(self):
        self.clear(1, 10)
        self.assertEqual(self.guard.update(packet(2, 10), 10, 10.1).state, "SIN YOLO")

    def test_recovery_after_dropout_restarts_clear_hold(self):
        self.clear(1, 10)
        self.clear(2, 10.7)
        self.assertTrue(self.clear(3, 11.3).allowed)
        self.assertFalse(self.clear(4, 13).allowed)
        self.assertFalse(self.clear(5, 13.7).allowed)
        self.assertTrue(self.clear(6, 14.3).allowed)

    def test_invalid_detection_cannot_be_treated_as_empty_list(self):
        for boxes in (
            None,
            {},
            [None],
            [dict(label="cat", confidence=0.9)],
            [dict(label="dog", confidence=float("nan"))],
        ):
            with self.subTest(boxes=boxes):
                self.assertEqual(
                    self.guard.update(packet(detections=boxes), 10, 10).state,
                    "SIN YOLO",
                )


if __name__ == "__main__":
    unittest.main()
