from unittest.mock import MagicMock, patch

from graph_bot.utils.slack import send_slack_notification


def test_send_slack_notification_no_url():
    with patch("urllib.request.urlopen") as mock_urlopen:
        send_slack_notification(None, {"text": "hello"})
        mock_urlopen.assert_not_called()


def test_send_slack_notification_success():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        send_slack_notification("http://example.com", {"text": "hello"})

        mock_urlopen.assert_called_once()
        # Verify payload
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == "http://example.com"
        assert req.data == b'{"text": "hello"}'


def test_send_slack_notification_failure_log():
    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("graph_bot.utils.slack.logger") as mock_logger,
    ):
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.reason = "Internal Server Error"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        send_slack_notification("http://example.com", {"text": "hello"})

        mock_logger.error.assert_called_with(
            "Failed to send Slack notification: 500 Internal Server Error"
        )
