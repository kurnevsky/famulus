# Famulus: AI-Powered LSP Server

![Logo](./logo.png)

## Introduction

Famulus, the Latin word for "servant" or "apprentice," refers to a magical
attendant who assists a powerful sorcerer in their spells and incantations. As
an LSP (Language Server Protocol) server, Famulus is designed to integrate Large
Language Models (LLMs) for code completions and code actions, and will be a
trusted companion to help you with your sorcery!

## Features

- [x] **Inline Code Completions**: Get context-aware code suggestions using LLMs
  - [x] [llama.cpp infill API](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#post-infill-for-code-infilling)
  - [x] [Ollama generate API](https://github.com/ollama/ollama/blob/v0.5.4/docs/api.md#generate-a-completion)
  - [x] [Mistral FIM API](https://docs.mistral.ai/api/#tag/fim)
  - [ ] [OpanAI completions API](https://platform.openai.com/docs/api-reference/completions)
- [ ] **Code Actions**: Automate routine tasks, such as code refactoring

## Getting Started

You can install Famulus using `cargo install famulus`. To use Famulus with your
preferred code editor, please consult your editor's documentation to learn how
to connect to an LSP server.

## Configuration

Configuration for Famulus is done via the `initializationOptions` object, which
is passed to the server during initialization. This object allows you to customize
the behavior of Famulus, such as specifying the provider for code completions
and configuring its settings.

The configuration options for each provider are as follows:

### LlamaCpp

```json
{
  "infill": {
    "provider": "LlamaCpp",
    "config": {
      "url": "http://localhost:8080/infill",
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

API key can be specified via `LLAMA_CPP_API_KEY` environment variable.

### Ollama

```json
{
  "infill": {
    "provider": "Ollama",
    "config": {
      "url": "http://localhost:11434/api/generate",
      "model": "qwen2.5-coder",
      "temperature": 0.7,
      "num_predict": 1024,
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

API key can be specified via `OLLAMA_API_KEY` environment variable. Note that
Ollama does not natively support the Bearer authentication scheme. However, this
functionality can be added using a reverse proxy.

### Mistral

```json
{
  "infill": {
    "provider": "Mistral",
    "config": {
      "url": "https://api.mistral.ai/v1/fim/completions",
      "model": "codestral-latest",
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 1024,
      "min_tokens": 1,
      "stop": ["\n\n"],
      "random_seed": 42
    }
  }
}
```

API key can be specified via `MISTRAL_API_KEY` environment variable.

## Contributing

We welcome contributions to Famulus! If you're interested in helping out, please:

1. **Fork** this repository
2. **Create** a new branch for your feature or bug fix
3. **Submit** a pull request with a detailed description of your changes

## License

Famulus is licensed under the [AGPL-3.0+ License](./LICENSE.txt). By
contributing to or using Famulus, you agree to the terms and conditions of this license.
