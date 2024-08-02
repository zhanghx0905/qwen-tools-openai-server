
# Qwen OpenAI API Server with ReAct Tools

## Introduction

This is an OpenAI API-compatible server based on the Qwen model, implementing tool calling functionality using the ReAct strategy. Qwen, a large language model developed by Alibaba Cloud, is equipped with the capability to call tools. However, common inference engines such as vLLM and sglang do not support tools APIs.

The purpose of this project is to provide an interface that is compatible with the OpenAI API, enabling developers to easily integrate Qwen's tool calling capabilities into their applications.

## Installation

1. Clone or download this project to your local machine.
2. Install the required libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the server:

```bash
python openai_api.py
```

By default, the server will run on `http://localhost:8600`.

## Usage

You can call the API using `curl` or any HTTP client (such as Postman). For example:

```bash
curl -X 'POST' \
  'http://localhost:8600/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "What\'s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto", 
  "stream":true
}'
```

---

## 简介

这是一个基于Qwen模型的OpenAI API格式的服务器，用 ReAct 策略实现了（流式）工具调用功能。

Qwen是阿里云开发的大型语言模型，具备工具调用能力。但是常见的推理引擎，如 vLLM，sglang 均不支持 tools API。

本项目旨在提供一个兼容OpenAI API的接口，使得开发者可以轻松地集成Qwen的工具调用能力。

## 安装

1. 克隆或下载本项目到本地。
2. 使用`pip`安装依赖库：

   ```bash
   pip install -r requirements.txt
   ```

## 运行

启动服务器：

```bash
python openai_api.py
```

默认情况下，服务器将在本地的`http://localhost:8600`上运行。

## 使用

你可以使用`curl`或任何HTTP客户端（如Postman）来调用API。例如：

```bash
curl -X 'POST' \
  'http://localhost:8600/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto", 
  "stream":true
}'
```

