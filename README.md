# Famulus: AI-Powered LSP Server

![Logo](./logo.png)

## Introduction

Famulus, the Latin word for "servant" or "apprentice," refers to a magical
attendant who assists a powerful sorcerer in their spells and incantations. As
an LSP (Language Server Protocol) server, Famulus is designed to integrate Large
Language Models (LLMs) for code completions and code actions, and will be a
trusted companion to help you with your sorcery!

## Features

- **Inline Code Completions**: Get context-aware code suggestions using LLMs
  - [llama.cpp infill API](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#post-infill-for-code-infilling)
  - [Ollama generate API](https://github.com/ollama/ollama/blob/v0.5.4/docs/api.md#generate-a-completion)
  - [Mistral FIM API](https://docs.mistral.ai/api/#tag/fim)
  - [OpanAI completions API](https://platform.openai.com/docs/api-reference/completions)

## Getting Started

You can install Famulus using `cargo install famulus`. To use Famulus with your
preferred code editor, please consult your editor's documentation to learn how
to connect to an LSP server.

## Configuration

**WARNING**: Configuration is currently unstable. Breaking changes are expected
to occur between versions.

Configuration for Famulus is done via the `initializationOptions` object, which
is passed to the server during initialization. This object allows you to customize
the behavior of Famulus, such as specifying the provider for code completions
and configuring its settings.

### Inline completion

The configuration options for each provider are as follows:

#### LlamaCpp

```json
{
  "infill": {
    "provider": "LlamaCpp",
    "config": {
      "url": "http://localhost:8080/infill",
      "api_key_env": "LLAMA_CPP_API_KEY",
      "temperature": 0.7,
      "max_tokens": 1024,
      "stop": ["<|file_separator|>"],
      "seed": 42
    }
  }
}
```

In order to use llama-cpp you will need a running server. You can launch it
using the following command:

```sh
llama-server -m codegemma-2b-Q6_K.gguf -c 0 -t 8 -ngl 19 -fa --port 8080
```

#### Ollama

```json
{
  "infill": {
    "provider": "Ollama",
    "config": {
      "url": "http://localhost:11434/api/generate",
      "api_key_env": "OLLAMA_API_KEY",
      "model": "qwen2.5-coder",
      "temperature": 0.7,
      "max_tokens": 1024,
      "stop": [],
      "seed": 42
    }
  }
}
```

In order to use Ollama you will need a running server. You can launch it
using the following command:

```sh
ollama serve
```

Ollama does not natively support the Bearer authentication scheme. However, this
functionality can be added using a reverse proxy.

#### Mistral

```json
{
  "infill": {
    "provider": "Mistral",
    "config": {
      "url": "https://api.mistral.ai/v1/fim/completions",
      "api_key_env": "MISTRAL_API_KEY",
      "model": "codestral-latest",
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 1024,
      "min_tokens": 1,
      "stop": ["\n\n"],
      "seed": 42
    }
  }
}
```

#### OpanAI completions

If your provider doesn't support the infill API, you can use the OpenAI completions
API instead. This allows you to manually specify a template that the model can use
to handle infill requests.

```json
{
  "infill": {
    "provider": "OpenAICompletions",
    "config": {
      "url": "http://localhost:8080/v1/completions",
      "api_key_env": "OPENAI_API_KEY",
      "model": "qwen2.5-coder",
      "temperature": 0.7,
      "max_tokens": 1024,
      "stop": [],
      "seed": 42
    },
    "template": "<|fim_prefix|>{{ prefix }}<|fim_suffix|>{{ suffix }}<|fim_middle|>"
  }
}
```

## Contributing

We welcome contributions to Famulus! If you're interested in helping out, please:

1. **Fork** this repository
2. **Create** a new branch for your feature or bug fix
3. **Submit** a pull request with a detailed description of your changes

## License

Famulus is licensed under the [AGPL-3.0+ License](./LICENSE.txt). By
contributing to or using Famulus, you agree to the terms and conditions of this
license.
