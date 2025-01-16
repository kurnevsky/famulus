use std::env;

use either::Either;
use serde::Deserialize;

use crate::{
  chat::Chat,
  infill::Infill,
  llama_cpp::infill::{LlamaCppInfill, LlamaCppInfillConfig},
  mistral::infill::{MistralInfill, MistralInfillConfig},
  ollama::infill::{OllamaInfill, OllamaInfillConfig},
  openai::chat::{OpenAIChat, OpenAIChatConfig},
};

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
#[serde(tag = "privider", content = "config")]
pub enum CompletionConfig {
  #[default]
  Empty,
  Mistral(MistralInfillConfig),
  LlamaCpp(LlamaCppInfillConfig),
  Ollama(OllamaInfillConfig),
}

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
#[serde(tag = "privider", content = "config")]
pub enum ChatConfig {
  #[default]
  Empty,
  OpenAI(OpenAIChatConfig),
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct Config {
  #[serde(default)]
  infill: CompletionConfig,
  #[serde(default)]
  chat: ChatConfig,
}

impl Config {
  pub fn get_infill(self) -> (impl Infill + Clone + Send, impl Chat + Clone + Send) {
    let infill = match self.infill {
      CompletionConfig::Empty => Either::Left(Either::Left(())),
      CompletionConfig::Mistral(config) => Either::Left(Either::Right(MistralInfill {
        api_key: env::var("MISTRAL_API_KEY").unwrap(),
        config,
      })),
      CompletionConfig::LlamaCpp(config) => Either::Right(Either::Left(LlamaCppInfill {
        api_key: env::var("LLAMA_CPP_API_KEY").ok(),
        config,
      })),
      CompletionConfig::Ollama(config) => Either::Right(Either::Right(OllamaInfill {
        api_key: env::var("OLLAMA_API_KEY").ok(),
        config,
      })),
    };
    let chat = match self.chat {
      ChatConfig::Empty => Either::Left(()),
      ChatConfig::OpenAI(config) => Either::Right(OpenAIChat { config }),
    };
    (infill, chat)
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    config::ChatConfig, llama_cpp::infill::LlamaCppInfillConfig, mistral::infill::MistralInfillConfig,
    ollama::infill::OllamaInfillConfig,
  };

  use super::{CompletionConfig, Config};

  #[test]
  fn mistral_config() {
    let str = r#"
    {
      "infill": {
        "privider": "Mistral",
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
    "#;
    let config = Config {
      infill: CompletionConfig::Mistral(MistralInfillConfig {
        url: "https://api.mistral.ai/v1/fim/completions".to_string(),
        model: "codestral-latest".to_string(),
        temperature: Some(0.7),
        top_p: Some(0.95),
        max_tokens: Some(1024),
        min_tokens: Some(1),
        stop: vec!["\n\n".to_string()],
        random_seed: Some(42),
      }),
      chat: ChatConfig::Empty,
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }

  #[test]
  fn llama_cpp_config() {
    let str = r#"
    {
      "infill": {
        "privider": "LlamaCpp",
        "config": {
          "url": "http://localhost:8080/infill",
          "temperature": 0.7,
          "max_tokens": 1024,
          "stop": ["<|file_separator|>"],
          "seed": 42
        }
      }
    }
    "#;
    let config = Config {
      infill: CompletionConfig::LlamaCpp(LlamaCppInfillConfig {
        url: "http://localhost:8080/infill".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1024),
        stop: vec!["<|file_separator|>".to_string()],
        seed: Some(42),
      }),
      chat: ChatConfig::Empty,
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }

  #[test]
  fn ollama_config() {
    let str = r#"
     {
       "infill": {
         "privider": "Ollama",
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
     "#;
    let config = Config {
      infill: CompletionConfig::Ollama(OllamaInfillConfig {
        url: "http://localhost:11434/api/generate".to_string(),
        model: "qwen2.5-coder".to_string(),
        temperature: Some(0.7),
        stop: vec![],
        num_predict: Some(1024),
        seed: Some(42),
      }),
      chat: ChatConfig::Empty,
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }
}
