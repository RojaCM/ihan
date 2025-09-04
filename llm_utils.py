import json
import os
import logging
from slip.auth import get_auth
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from typing import List, Optional, Any, Tuple
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.outputs import ChatResult

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

AVAILABLE_MODELS = {
    # OpenAI models
    "gpt-35-turbo-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-35-turbo-16k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-8k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-32k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-turbo-1106-preview": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-turbo-0613":{"provider": "OpenAI", "type": "chat"},
    "gpt-35-turbo-0125": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-turbo-0125-preview": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-turbo-0409": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-vision-preview": {"provider": "OpenAI", "type": "chat"},
    "gpt-4o-0513": {"provider": "OpenAI", "type": "chat"},
    "gpt-4o-0806": {"provider": "OpenAI", "type": "chat"},
    
    # Llama models
    "meta-llama/Llama-2-7b-chat-hf": {"provider": "SLIP", "type": "chat"},
    "meta-llama/Llama-2-70b-chat-hf": {"provider": "SLIP", "type": "chat"},
    "meta-llama/Meta-Llama-3-70B-Instruct": {"provider": "SLIP", "type": "Instruct"},
    "meta-llama/Meta-Llama-3-8B-Instruct": {"provider": "SLIP", "type": "Instruct"},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"provider": "SLIP", "type": "Instruct"},
    # Mistral Models
    "mistralai/Mistral-7B-Instruct-v0.2": {"provider": "SLIP", "type": "instruct"},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"provider": "SLIP", "type": "instruct"},
    # Zephyr
    "HuggingFaceH4/zephyr-7b-beta": {"provider": "SLIP", "type": "instruct"},
}


def _patched_generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    stream: Optional[bool] = None,
    **kwargs: Any,
) -> ChatResult:
    should_stream = stream if stream is not None else self.streaming
    if should_stream:
        stream_iter = self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return generate_from_stream(stream_iter)
    message_dicts, params = self._create_message_dicts(messages, stop)
    params = {
        **params,
        **({"stream": stream} if stream is not None else {}),
        **kwargs,
    }
    response = self.client.create(messages=message_dicts, **params)
    # patching - transform the string typed response from out LLM gateway into a dict
    if isinstance(response, str):
        response = json.loads(response)
    return self._create_chat_result(response)


async def _patched_agenerate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    stream: Optional[bool] = None,
    **kwargs: Any,
) -> ChatResult:
    should_stream = stream if stream is not None else self.streaming
    if should_stream:
        stream_iter = self._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await agenerate_from_stream(stream_iter)

    message_dicts, params = self._create_message_dicts(messages, stop)
    params = {
        **params,
        **({"stream": stream} if stream is not None else {}),
        **kwargs,
    }
    response = await self.async_client.create(messages=message_dicts, **params)
    # patching - transform the string typed response from out LLM gateway into a dict
    if isinstance(response, str):
        response = json.loads(response)
    return self._create_chat_result(response)


# patch the ChatOpenAI object
ChatOpenAI._generate = _patched_generate
ChatOpenAI._agenerate = _patched_agenerate


def get_llm(model_tag: str, max_token=500, temperature=0) -> ChatOpenAI:
    """
    Return a ChatOpenAI object, alters the API Key and URL based on whether it's a OpenAI based model or a model available on SLIP
    Currently, the available models are: 
    # OpenAI models
    "gpt-35-turbo-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-35-turbo-16k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-8k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-32k-0613": {"provider": "OpenAI", "type": "chat"},
    "gpt-4-turbo-1106-preview": {"provider": "OpenAI", "type": "chat"},
    # Llama-2 models
    "meta-llama/Llama-2-7b-chat-hf": {"provider": "SLIP", "type": "chat"},
    "meta-llama/Llama-2-70b-chat-hf": {"provider": "SLIP", "type": "chat"},
    # Mistral Models
    "mistralai/Mistral-7B-Instruct-v0.2": {"provider": "SLIP", "type": "instruct"},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"provider": "SLIP", "type": "instruct"},
    # Zephyr
    "HuggingFaceH4/zephyr-7b-beta": {"provider": "SLIP", "type": "instruct"},
    """
    assert (
        model_tag in AVAILABLE_MODELS
    ), f"Model Tag is not one of the available models. Currently available models are {AVAILABLE_MODELS.keys()}"

    model_info = AVAILABLE_MODELS[model_tag]
    if model_info["provider"] == "OpenAI":
        return ChatOpenAI(model=model_tag, temperature=temperature)

    # if the model is not from OpenAI, we need to set up SLIP by logging into SLIP
    slip_auth = get_auth(
        login=True,
        url=os.environ["SLIP_AUTH_URL"],
        password=os.environ["SLIP_PASSWORD"],
        username=os.environ["SLIP_USERNAME"],
        client_id=os.environ["SLIP_CLIENT_ID"],
    )
    
    slip_auth.refresh()
    BASE_URL = os.environ["SLIP_BASE_URL"]
    CLIENT_URL = f"{BASE_URL}/openai/v1"
    SLIP_APP_NAME = os.environ.get("SLIP_APP_NAME", "slip-sandbox")
    
    return ChatOpenAI(
        model=model_tag,
        temperature=temperature,
        api_key=slip_auth.get_headers()["Authorization"].split("Bearer ")[-1],
        base_url=CLIENT_URL,
        default_headers={"Slip-App-Name": SLIP_APP_NAME},
        max_tokens=max_token,
        model_kwargs = {'seed':123456}
        
    )