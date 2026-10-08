"""Demo speech settings and gateway routes for Chainlit audio."""

import importlib
import json
from email.parser import BytesParser
from email.policy import HTTP
from unittest.mock import AsyncMock

import chainlit as cl
import httpx2
import pytest
from chainlit_utils.openai.audio import SPEECH_ACTION_NAME

from lgos_chainlit import audio
from tests.support import message, response, streamed, user_message


def _form(request: httpx2.Request) -> dict[str, bytes]:
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    form = BytesParser(policy=HTTP).parsebytes(head + request.content)
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in form.iter_parts()
    }


async def test_recording_is_transcribed_by_the_configured_gateway_model(
    monkeypatch: pytest.MonkeyPatch,
    chainlit_context,
    fake_gateway,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    fake_gateway.replies.append(httpx2.Response(200, json={"text": "What time is it?"}))
    window_message = AsyncMock()
    monkeypatch.setattr(chainlit_context.emitter, "send_window_message", window_message)

    await chat.on_audio_start()
    await chat.on_audio_chunk(
        cl.InputAudioChunk(
            isStart=True, mimeType="pcm16", elapsedTime=0, data=b"\x01\x00"
        )
    )
    await chat.on_audio_end()

    [transcription] = fake_gateway.requests
    assert transcription.url.path == "/v1/audio/transcriptions"
    assert _form(transcription)["model"] == b"openai/gpt-4o-mini-transcribe"
    assert window_message.await_args.args[0]["text"] == "What time is it?"


@pytest.mark.parametrize("speech", [True, False], ids=["speech", "no-speech"])
async def test_answer_gets_a_speech_button_when_a_speech_model_is_set(
    monkeypatch: pytest.MonkeyPatch,
    speech: bool,
    chainlit_context,
    fake_gateway,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    send_element = AsyncMock()
    monkeypatch.setattr(
        chainlit_context.session, "persist_file", AsyncMock(return_value={"id": "f"})
    )
    monkeypatch.setattr(chainlit_context.emitter, "send_element", send_element)
    if not speech:
        monkeypatch.setattr(audio.settings, "AUDIO_TTS_MODEL", None)
    chainlit_context.session.chat_profile = "lgos-a/simple-graph"
    fake_gateway.replies.append(streamed(response(message("Paris."))))

    await chat.on_message(user_message("Capital of France?"))

    answer = cl.chat_context.get()[-1]
    elements = [call.args[0] for call in send_element.await_args_list]
    assert answer.content == "Paris."
    assert [(element["name"], element["forId"]) for element in elements] == (
        [("SpeechButton", answer.id)] if speech else []
    )


@pytest.mark.parametrize("speech", [True, False], ids=["speech", "no-speech"])
async def test_read_aloud_uses_the_configured_gateway_model_and_voice(
    monkeypatch: pytest.MonkeyPatch,
    speech: bool,
    chainlit_context,
    fake_gateway,
) -> None:
    chat = importlib.import_module("lgos_chainlit.chat")
    answer = cl.chat_context.add(cl.Message(content="It is noon in Paris."))
    if not speech:
        monkeypatch.setattr(audio.settings, "AUDIO_TTS_MODEL", None)
    fake_gateway.replies.append(
        httpx2.Response(
            200, content=b"mp3-bytes", headers={"content-type": "audio/mpeg"}
        )
    )

    result = await chat.on_speech_request(
        cl.Action(
            name=SPEECH_ACTION_NAME,
            payload={"message_id": answer.id, "part": 0},
        )
    )

    if not speech:
        assert result == {"ok": False, "error": "This answer cannot be read aloud."}
        assert fake_gateway.requests == []
        return
    assert result["ok"] is True
    [request] = fake_gateway.requests
    assert request.url.path == "/v1/audio/speech"
    assert json.loads(request.content) == {
        "model": "openai/gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "It is noon in Paris.",
    }
