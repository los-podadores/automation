"""Manual control semantics: no unexpected start or restart after a lost lease."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_control import RouteOperator


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.demo = SimpleNamespace(enabled=False, finished=False, failure="")
        self.demo.stop = Mock(side_effect=lambda: setattr(self.demo, "enabled", False))
        self.operator = RouteOperator(Path("unused"))

    def command(self, action="start", request="request-1", stamp=10):
        return dict(
            action=action,
            request_id=request,
            request_order=int(request.split("-")[-1]),
            written_at=stamp,
        )

    def test_waits_for_explicit_start_and_accepts_stop(self):
        self.assertFalse(self.operator.update(self.demo, None, 10))
        self.assertFalse(self.demo.enabled)
        self.assertTrue(self.operator.update(self.demo, self.command(), 10))
        self.assertTrue(self.demo.enabled)
        self.operator.update(self.demo, self.command("stop", "request-2"), 10)
        self.assertFalse(self.demo.enabled)

    def test_lost_heartbeat_stops_and_old_start_cannot_resume(self):
        self.operator.update(self.demo, self.command(), 10)
        self.assertFalse(self.operator.update(self.demo, self.command(), 13))
        self.assertFalse(self.demo.enabled)
        self.operator.update(self.demo, self.command(stamp=13), 13)
        self.assertFalse(self.demo.enabled)
        self.operator.update(self.demo, self.command(request="request-2", stamp=13), 13)
        self.assertTrue(self.demo.enabled)

    def test_completed_or_failed_route_cannot_restart(self):
        for finished, failure in ((True, ""), (False, "fuera del camino")):
            with self.subTest(finished=finished, failure=failure):
                self.demo.finished, self.demo.failure = finished, failure
                self.operator.last_request = None
                self.operator.last_order = -1
                self.operator.update(self.demo, self.command(), 10)
                self.assertFalse(self.demo.enabled)

    def test_transient_file_lock_retains_only_unexpired_heartbeat(self):
        with (
            patch(
                "session_control.read_json", side_effect=[self.command(), None, None]
            ),
            patch("session_control.time.time", side_effect=[10, 10.1, 13]),
        ):
            self.assertTrue(self.operator.poll(self.demo))
            self.assertTrue(self.demo.enabled)
            self.assertTrue(self.operator.poll(self.demo))
            self.assertTrue(self.demo.enabled)
            self.assertFalse(self.operator.poll(self.demo))
            self.assertFalse(self.demo.enabled)

    def test_invalid_operator_data_stops(self):
        for payload in (
            None,
            {},
            self.command(action="drive"),
            self.command(stamp=float("nan")),
            self.command(stamp=100),
            self.command() | {"request_order": "invalid"},
        ):
            self.demo.enabled = True
            self.assertFalse(self.operator.update(self.demo, payload, 10))
            self.assertFalse(self.demo.enabled)

    def test_poll_rejects_invalid_order_without_crashing(self):
        self.demo.enabled = True
        with patch(
            "session_control.read_json",
            return_value=self.command() | {"request_order": "invalid"},
        ):
            self.assertFalse(self.operator.poll(self.demo))
        self.assertFalse(self.demo.enabled)

    def test_cached_old_stop_does_not_override_new_start(self):
        self.operator.update(self.demo, self.command("stop", "request-1"), 10)
        self.operator.update(self.demo, self.command("start", "request-2"), 10)
        self.assertTrue(self.demo.enabled)
        self.operator.update(self.demo, self.command("stop", "request-1"), 10)
        self.assertTrue(self.demo.enabled)
        self.operator.update(self.demo, self.command("stop", "request-3"), 10)
        self.assertFalse(self.demo.enabled)


if __name__ == "__main__":
    unittest.main()
