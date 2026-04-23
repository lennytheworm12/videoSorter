import unittest
from unittest import mock

from core.youtube_network import YouTubeNetworkPolicy


class YouTubeNetworkPolicyTests(unittest.TestCase):
    def test_direct_success_never_uses_proxy(self) -> None:
        policy = YouTubeNetworkPolicy(proxy_urls=["http://proxy"], direct_retry_delays=[1], proxy_retry_delays=[1])
        seen: list[str | None] = []

        def operation(proxy_url: str | None) -> str:
            seen.append(proxy_url)
            return "ok"

        result = policy.run("test", operation)
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [None])

    def test_direct_rate_limit_retries_direct_before_proxy(self) -> None:
        policy = YouTubeNetworkPolicy(proxy_urls=["http://proxy"], direct_retry_delays=[5, 10], proxy_retry_delays=[20])
        seen: list[str | None] = []
        failures = iter([Exception("429 too many requests"), Exception("429 too many requests")])

        def operation(proxy_url: str | None) -> str:
            seen.append(proxy_url)
            try:
                raise next(failures)
            except StopIteration:
                return "ok"

        with mock.patch("core.youtube_network.time.sleep") as sleep_mock:
            result = policy.run("test", operation)

        self.assertEqual(result, "ok")
        self.assertEqual(seen, [None, None, None])
        sleep_mock.assert_has_calls([mock.call(5), mock.call(10)])

    def test_direct_exhaustion_switches_to_proxy(self) -> None:
        policy = YouTubeNetworkPolicy(proxy_urls=["http://proxy-a"], direct_retry_delays=[5, 10], proxy_retry_delays=[20])
        seen: list[str | None] = []
        state = {"calls": 0}

        def operation(proxy_url: str | None) -> str:
            seen.append(proxy_url)
            state["calls"] += 1
            if state["calls"] <= 3:
                raise Exception("429 too many requests")
            return "ok"

        with mock.patch("core.youtube_network.time.sleep"):
            with mock.patch("core.youtube_network.time.time", return_value=100):
                result = policy.run("test", operation)

        self.assertEqual(result, "ok")
        self.assertEqual(seen[:3], [None, None, None])
        self.assertEqual(seen[3], "http://proxy-a")
        self.assertGreater(policy.local_blocked_until, 100)

    def test_blocked_direct_uses_proxy_immediately(self) -> None:
        policy = YouTubeNetworkPolicy(proxy_urls=["http://proxy-a"], direct_retry_delays=[5], proxy_retry_delays=[20])
        policy.local_blocked_until = 200
        seen: list[str | None] = []

        def operation(proxy_url: str | None) -> str:
            seen.append(proxy_url)
            return "ok"

        with mock.patch("core.youtube_network.time.time", return_value=100):
            result = policy.run("test", operation)

        self.assertEqual(result, "ok")
        self.assertEqual(seen, ["http://proxy-a"])


if __name__ == "__main__":
    unittest.main()
