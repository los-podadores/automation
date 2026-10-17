"""Manual control semantics: no unexpected start or restart after a lost lease."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_control import PacketReceiver, RouteOperator


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
            heartbeat_order=int(stamp * 1000) + int(request.split("-")[-1]),
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
            patch("session_control.time.monotonic", side_effect=[10, 10.1, 13]),
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
            self.command() | {"written_at": float("nan")},
            self.command() | {"request_order": "invalid"},
            self.command() | {"heartbeat_order": "invalid"},
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

    def test_windows_wsl_clock_skew_does_not_cancel_start(self):
        # Real packet times can be ahead of, or behind, the other OS clock.
        command = self.command() | {"written_at": 999999.0}
        self.assertTrue(self.operator.update(self.demo, command, 10))
        command = self.command(stamp=11) | {"written_at": -999999.0}
        self.assertTrue(self.operator.update(self.demo, command, 11))
        self.assertTrue(self.demo.enabled)
        self.assertFalse(self.operator.update(self.demo, command, 13.1))
        self.assertFalse(self.demo.enabled)

    def test_status_packets_expire_when_their_sequence_freezes(self):
        receiver = PacketReceiver("status_order")
        packet = dict(status_order=100, written_at=999999.0)
        self.assertEqual(receiver.receive(packet, 10), packet)
        self.assertEqual(receiver.receive(packet, 11), packet)
        self.assertIsNone(receiver.receive(packet, 12.1))
        self.assertIsNone(receiver.receive(dict(status_order=99), 12.2))
        self.assertIsNotNone(receiver.receive(dict(status_order=101), 12.3))


if __name__ == "__main__":
    unittest.main()
