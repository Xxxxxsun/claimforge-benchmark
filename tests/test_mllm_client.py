import json
import unittest
from unittest import mock

from eval.mllm.client import RetryableError, VisionClient


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def _gemini_model():
    return {
        "id": "gemini-3.1-pro-preview",
        "slug": "gemini",
        "requestFormat": "openai_chat_completions",
        "maxTokens": 2000,
        "temperature": 0,
        "omitTemperature": True,
        "provider": {
            "apiBase": "https://gateway.example/openai",
            "apiKey": "test-key",
            "extraBody": {
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
            },
        },
    }


class GeminiOpenAICompatibilityTest(unittest.TestCase):
    def test_uses_standard_multimodal_payload_with_low_reasoning_json_output(self):
        client = VisionClient(_gemini_model(), 120, {"detail": "high"})
        response = _Response({
            "choices": [{
                "message": {
                    "content": '{"decision":"not_edited"}',
                },
            }],
        })

        with mock.patch(
            "eval.mllm.client.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            content, _ = client.call(
                "system",
                "inspect this image",
                "data:image/png;base64,AA==",
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(content, '{"decision":"not_edited"}')
        self.assertEqual(payload["model"], "gemini-3.1-pro-preview")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", payload)
        self.assertNotIn("prompt", payload)
        self.assertEqual(
            payload["messages"][1]["content"][1],
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AA==",
                    "detail": "high",
                },
            },
        )

    def test_connection_reset_is_retryable(self):
        client = VisionClient(_gemini_model(), 120, {"detail": "high"})

        with mock.patch(
            "eval.mllm.client.urllib.request.urlopen",
            side_effect=ConnectionResetError("connection reset"),
        ):
            with self.assertRaises(RetryableError):
                client.call(
                    "system",
                    "inspect this image",
                    "data:image/png;base64,AA==",
                )


if __name__ == "__main__":
    unittest.main()
