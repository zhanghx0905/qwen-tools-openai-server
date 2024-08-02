import json
import os
import uuid
from typing import Any, List, Optional, Union

import openai
import orjson
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel

app = FastAPI()

# 设置Qwen OpenAI API的URL和密钥
load_dotenv(override=True)
OPENAI_BASEURL = os.getenv("OPENAI_BASEURL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT = openai.AsyncOpenAI(base_url=OPENAI_BASEURL, api_key=OPENAI_API_KEY)


class ChatCompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    tools: Optional[list[ChatCompletionToolParam]] = None
    stop: Union[str, List[str], None] = None


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""


def process_tools_from_prompt(
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam] | None,
) -> list[ChatCompletionMessageParam]:
    if tools:
        tools_text = []
        tools_name_text = []
        for func_info in tools:
            parameters = []
            fp = func_info["function"].get("parameters", {})
            if fp:
                required_parameters = fp.get("required", [])
                for name, p in fp["properties"].items():  # type: ignore
                    param = dict({"name": name}, **p)
                    if name in required_parameters:
                        param["required"] = True
                    parameters.append(param)

            name = func_info["function"]["name"]
            desc = func_info["function"].get("description", "")
            tool_string = TOOL_DESC.format(
                name_for_model=name,
                name_for_human=name,
                description_for_model=desc,
                parameters=json.dumps(parameters, ensure_ascii=False),
            )
            tools_text.append(tool_string)
            tools_name_text.append(name)
        tools_text_string = "\n\n".join(tools_text)
        tools_name_text_string = ", ".join(tools_name_text)
        tool_system = REACT_INSTRUCTION.format(
            tools_text=tools_text_string,
            tools_name_text=tools_name_text_string,
        )
    else:
        tool_system = ""

    new_messages = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if tools:
            if role == "user":
                if tool_system:
                    content = tool_system + f"\n\nQuestion: {content}"
                    tool_system = ""
                else:
                    content = f"Question: {content}"
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    func_call = tool_calls[0]["function"]
                    f_name, f_args = (
                        func_call["name"],
                        func_call["arguments"],
                    )
                    content = f"Thought: I can use {f_name}.\nAction: {f_name}\nAction Input: {f_args}"
                elif content:
                    content = f"Thought: I now know the final answer.\nFinal Answer: {content}"
            elif role == "tool":
                content = f"Observation: {content}"
        if content:
            content = content.lstrip("\n").rstrip()
        new_messages.append({"role": role, "content": content})

    return new_messages


def eval_qwen_tools_arguments(text: str, tool_names: set[str]):
    """
    解析文本以提取函数名、函数参数和内容。
    """
    try:
        func_name, func_args, content = "", "", ""
        i = text.rfind("\nAction:")
        j = text.rfind("\nAction Input:")
        k = text.rfind("\nObservation:")
        t = max(text.rfind("\nThought:", 0, i), text.rfind("Thought:", 0, i))
        if 0 <= i < j:
            if k < j:
                text = text.rstrip() + "\nObservation:"
                k = text.rfind("\nObservation:")
        if 0 <= t < i < j < k:
            func_name = text[i + len("\nAction:") : j].strip()
            func_args = text[j + len("\nAction Input:") : k].strip()
            content = text[t + len("\nThought:") : i].strip()
        if func_name in tool_names:
            return content, func_name, func_args
    except Exception as e:
        logger.error("Eval tool calls completion failed:", e)
    t = max(text.rfind("\nThought:"), text.rfind("Thought:"))
    z = max(text.rfind("\nFinal Answer:"), text.rfind("Final Answer:"))
    if z >= 0:
        text = text[z + len("\nFinal Answer:") :]
    else:
        text = text[t + len("\nThought:") :]
    return text, None, None


def tokens_filter():
    found = False

    def process_tokens(tokens: str, delta: str):
        nonlocal found
        # Once "Final Answer:" is found, future tokens are allowed.
        if found:
            return delta
        # Check if the token ends with "\nFinal Answer:" and update `found`.
        final_answer_idx = tokens.lower().rfind("\nfinal answer:")
        if final_answer_idx != -1:
            found = True
            return tokens[final_answer_idx + len("\nfinal answer:") :]
        return ""

    return process_tokens


@app.post("/v1/chat/completions")
async def proxy_completions(request: ChatCompletionRequest):
    logger.info(request)
    request.temperature = min(max(request.temperature, 0.1), 2)
    tools = request.tools
    stream = request.stream
    request.messages = process_tools_from_prompt(request.messages, tools)

    body = request.model_dump(exclude_none=True)
    tool_names = set()
    if tools:
        del body["tools"]
        tool_names = set(func["function"]["name"] for func in tools)
        # FIXME: 有的推理引擎不支持设置 stop
        add_stop_words(request, body)
    try:

        if stream:
            response_stream = await CLIENT.chat.completions.create(**body)

            async def stream_generator():
                previous_text: str = ""
                tools_token_filter = tokens_filter()
                async for i in response_stream:
                    chunk = i.model_dump(exclude_none=True)
                    choice = chunk["choices"][0]
                    delta = choice["delta"]
                    delta_content = delta.get("content", "")
                    previous_text += delta_content
                    if tools:
                        if (
                            previous_text.endswith("Observation:")
                            or choice.get("finish_reason") is not None
                        ):
                            _, func, args = eval_qwen_tools_arguments(
                                previous_text, tool_names
                            )
                            delta["content"] = tools_token_filter(
                                tokens=previous_text, delta=delta_content
                            )
                            if func is not None:
                                _id = str(uuid.uuid4())
                                delta["content"] = None
                                choice["finish_reason"] = "tool_calls"
                                delta["tool_calls"] = [
                                    {
                                        "id": _id,
                                        "function": {
                                            "name": func,
                                            "arguments": args,
                                        },
                                        "type": "function",
                                    }
                                ]
                                yield "data: " + orjson.dumps(chunk).decode() + "\n"
                                break
                        else:
                            # 过滤掉不需要的内容
                            delta["content"] = tools_token_filter(
                                tokens=previous_text, delta=delta_content
                            )
                            if not delta["content"]:
                                continue

                    yield "data: " + orjson.dumps(chunk).decode() + "\n"
                yield "data: [DONE]\n"

            # 在上下文中返回流式响应
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = await CLIENT.chat.completions.create(**body)
            ret = response.model_dump(exclude_none=True)

            # 处理返回的响应
            if tools:
                content, func, args = eval_qwen_tools_arguments(text, tool_names)
                choice["message"]["content"] = content
                if func and args:
                    _id = str(uuid.uuid4())
                    
                    choice["message"]["tool_calls"] = [
                        {
                            "id": _id,
                            "function": {"name": func, "arguments": args},
                            "type": "function",
                        }
                    ]
                    choice["finish_reason"] = "tool_calls"
            return JSONResponse(ret)

    except Exception as e:
        # 捕获HTTP异常并返回错误信息
        raise HTTPException(status_code=500, detail=str(e))


def add_stop_words(request, body):
    stop = request.stop
    if isinstance(stop, str):
        body["stop"] = [stop, "Observation:"]
    elif isinstance(stop, list):
        body["stop"] = list(stop) + ["Observation:"]
    else:
        body["stop"] = "Observation:"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8600)
