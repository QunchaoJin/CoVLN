import os
import base64
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def _build_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (and optionally OPENAI_BASE_URL for "
            "an OpenAI-compatible endpoint) before running, see README.md."
        )
    return OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))


client = _build_client()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def gpt_infer(system, other_index, text, image_list, other_list, model="gpt-4-vision-preview", max_tokens=600, response_format=None):
    """Query the MLLM with the agent's own node images plus the peer's shared node images.

    image_list: images of this agent's own map nodes (index i -> "Image i").
    other_list: images of the peer agent's map nodes.
    other_index: indices in other_list that are exposed to this agent ("Image other_i").
    """

    user_content = []
    for i, image in enumerate(image_list):
        if image is not None:
            user_content.append(
                {
                    "type": "text",
                    "text": f"Image {i}:"
                },
            )

            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            image_message = {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image_base64}",
                         "detail": "low"
                     }
                 }
            user_content.append(image_message)

    for i in other_index:
        if i >= len(other_list):
            continue
        other_path = other_list[i]
        if other_path is None:
            continue
        user_content.append(
            {
                "type": "text",
                "text": f"Image other_{i}:"
            },
        )
        with open(other_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low"
            }
        }

        user_content.append(image_message)

    user_content.append(
        {
            "type": "text",
            "text": text
        }
    )

    messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]

    if response_format:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
    else:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage

    return answer, tokens
