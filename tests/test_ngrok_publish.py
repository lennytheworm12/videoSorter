import unittest
from unittest import mock

from cloud import ngrok_publish


class NgrokPublishTests(unittest.TestCase):
    def test_https_tunnel_url_prefers_https_public_url(self) -> None:
        payload = {
            "tunnels": [
                {"proto": "http", "public_url": "http://plain.example.com"},
                {"proto": "https", "public_url": "https://secure.example.com"},
            ]
        }

        self.assertEqual(
            ngrok_publish._https_tunnel_url(payload),
            "https://secure.example.com",
        )

    def test_publish_current_url_marks_online_when_https_tunnel_exists(self) -> None:
        with mock.patch(
            "cloud.ngrok_publish._fetch_ngrok_payload",
            return_value={"tunnels": [{"proto": "https", "public_url": "https://strong.ngrok.dev"}]},
        ), mock.patch("cloud.ngrok_publish.upsert_runtime_config") as upsert:
            value = ngrok_publish.publish_current_url()

        self.assertTrue(value["online"])
        self.assertEqual(value["url"], "https://strong.ngrok.dev")
        upsert.assert_called_once()

    def test_publish_current_url_marks_offline_when_ngrok_is_unreachable(self) -> None:
        with mock.patch(
            "cloud.ngrok_publish._fetch_ngrok_payload",
            side_effect=OSError("ngrok unavailable"),
        ), mock.patch("cloud.ngrok_publish.upsert_runtime_config") as upsert:
            value = ngrok_publish.publish_current_url()

        self.assertFalse(value["online"])
        self.assertIsNone(value["url"])
        upsert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
