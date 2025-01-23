use std::{fmt::Debug, sync::Arc};

use either::Either;
use serde::{de::Error, Deserialize};

use crate::{chat::Chat, infill::Infill};

pub trait Provider {
  type Model: for<'a> Deserialize<'a> + Clone + PartialEq + Debug;
  type ApiKeyEnv: for<'a> Deserialize<'a> + Clone + PartialEq + Debug;
  type Temperature: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
  type TopP: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
  type MaxTokens: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
  type MinTokens: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
  type Stop: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
  type Seed: for<'a> Deserialize<'a> + Clone + PartialEq + Debug + Default;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Empty;
impl<'de> Deserialize<'de> for Empty {
  fn deserialize<D>(_: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    Err(Error::custom("Empty cannot be deserialized"))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub struct Mistral;
impl Provider for Mistral {
  type Model = String;
  type ApiKeyEnv = String;
  type Temperature = Option<f64>;
  type TopP = Option<f64>;
  type MaxTokens = Option<u32>;
  type MinTokens = Option<u32>;
  type Stop = Vec<String>;
  type Seed = Option<u32>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub struct LlamaCpp;
impl Provider for LlamaCpp {
  type Model = Option<Empty>;
  type ApiKeyEnv = Option<String>;
  type Temperature = Option<f64>;
  type TopP = Option<f64>;
  type MaxTokens = Option<u32>;
  type MinTokens = Option<Empty>;
  type Stop = Vec<String>;
  type Seed = Option<u32>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub struct Ollama;
impl Provider for Ollama {
  type Model = String;
  type ApiKeyEnv = Option<String>;
  type Temperature = Option<f64>;
  type TopP = Option<f64>;
  type MaxTokens = Option<u32>;
  type MinTokens = Option<Empty>;
  type Stop = Vec<String>;
  type Seed = Option<u32>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub struct OpenAI;
impl Provider for OpenAI {
  type Model = Option<String>;
  type ApiKeyEnv = Option<String>;
  type Temperature = Option<f64>;
  type TopP = Option<f64>;
  type MaxTokens = Option<u32>;
  type MinTokens = Option<Empty>;
  type Stop = Vec<String>;
  type Seed = Option<u32>;
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct GenerationConfig<P: Provider> {
  pub model: P::Model,
  #[serde(default)]
  pub temperature: P::Temperature,
  #[serde(default)]
  pub top_p: P::TopP,
  #[serde(default)]
  pub max_tokens: P::MaxTokens,
  #[serde(default)]
  pub min_tokens: P::MinTokens,
  #[serde(default)]
  pub stop: P::Stop,
  #[serde(default)]
  pub seed: P::Seed,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct ModelConfig<P: Provider> {
  pub url: String,
  #[serde(default)]
  pub api_key_env: P::ApiKeyEnv,
  #[serde(flatten)]
  pub generation_config: GenerationConfig<P>,
}

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
#[serde(tag = "provider", content = "config")]
pub enum CompletionConfig {
  #[default]
  Empty,
  Mistral(Arc<ModelConfig<Mistral>>),
  LlamaCpp(Arc<ModelConfig<LlamaCpp>>),
  Ollama(Arc<ModelConfig<Ollama>>),
}

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
#[serde(tag = "provider", content = "config")]
pub enum ChatModelConfig {
  #[default]
  Empty,
  OpenAI(Arc<ModelConfig<OpenAI>>),
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct MessageConfig {
  pub role: String,
  pub content: String,
}

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
pub struct ChatConfig {
  pub model_config: ChatModelConfig,
  pub messages: Vec<MessageConfig>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct Config {
  #[serde(default)]
  pub infill: CompletionConfig,
  #[serde(default)]
  pub chat: ChatConfig,
}

impl Config {
  pub fn get_infill(&self) -> impl Infill + Clone + Send {
    match self.infill {
      CompletionConfig::Empty => Either::Left(Either::Left(())),
      CompletionConfig::Mistral(ref config) => Either::Left(Either::Right(config.clone())),
      CompletionConfig::LlamaCpp(ref config) => Either::Right(Either::Left(config.clone())),
      CompletionConfig::Ollama(ref config) => Either::Right(Either::Right(config.clone())),
    }
  }

  pub fn get_chat(&self) -> impl Chat + Clone + Send {
    match self.chat.model_config {
      ChatModelConfig::Empty => Either::Left(()),
      ChatModelConfig::OpenAI(ref config) => Either::Right(config.clone()),
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use crate::config::{ChatConfig, CompletionConfig, Config, GenerationConfig, ModelConfig};

  #[test]
  fn mistral_infill_config() {
    let str = r#"
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
    "#;
    let config = Config {
      infill: CompletionConfig::Mistral(Arc::new(ModelConfig {
        url: "https://api.mistral.ai/v1/fim/completions".to_string(),
        api_key_env: "MISTRAL_API_KEY".to_string(),
        generation_config: GenerationConfig {
          model: "codestral-latest".to_string(),
          temperature: Some(0.7),
          top_p: Some(0.95),
          max_tokens: Some(1024),
          min_tokens: Some(1),
          stop: vec!["\n\n".to_string()],
          seed: Some(42),
        },
      })),
      chat: ChatConfig::default(),
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }

  #[test]
  fn llama_cpp_infill_config() {
    let str = r#"
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
    "#;
    let config = Config {
      infill: CompletionConfig::LlamaCpp(Arc::new(ModelConfig {
        url: "http://localhost:8080/infill".to_string(),
        api_key_env: Some("LLAMA_CPP_API_KEY".to_string()),
        generation_config: GenerationConfig {
          model: None,
          temperature: Some(0.7),
          top_p: None,
          max_tokens: Some(1024),
          min_tokens: None,
          stop: vec!["<|file_separator|>".to_string()],
          seed: Some(42),
        },
      })),
      chat: ChatConfig::default(),
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }

  #[test]
  fn ollama_infill_config() {
    let str = r#"
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
     "#;
    let config = Config {
      infill: CompletionConfig::Ollama(Arc::new(ModelConfig {
        url: "http://localhost:11434/api/generate".to_string(),
        api_key_env: Some("OLLAMA_API_KEY".to_string()),
        generation_config: GenerationConfig {
          model: "qwen2.5-coder".to_string(),
          temperature: Some(0.7),
          top_p: None,
          min_tokens: None,
          max_tokens: Some(1024),
          stop: vec![],
          seed: Some(42),
        },
      })),
      chat: ChatConfig::default(),
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }

  #[test]
  fn openai_chat_config() {
    let str = r#"
    {
      "chat": {
        "model_config": {
          "provider": "OpenAI",
          "config": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key_env": "OPENAI_API_KEY",
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stop": [],
            "seed": 42
          }
        },
        "messages": [
          {
            "role": "system",
            "content": "You provide the modified code directly without any surrounding explanation or context, and do not enclose it within a code block."
          },
          {
            "role": "user",
            "content": "{{ prompt }}\n\n```\n{{ selection }}\n```"
          }
        ]
      }
    }
    "#;
    let config = Config {
      infill: CompletionConfig::default(),
      chat: ChatConfig {
        model_config: super::ChatModelConfig::OpenAI(Arc::new(ModelConfig {
          url: "https://api.groq.com/openai/v1/chat/completions".to_string(),
          api_key_env: Some("OPENAI_API_KEY".to_string()),
          generation_config: GenerationConfig {
            model: Some("llama-3.3-70b-versatile".to_string()),
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(1024),
            min_tokens: None,
            stop: vec![],
            seed: Some(42),
          },
        })),
        messages: vec![super::MessageConfig {
          role: "system".to_string(),
          content: "You provide the modified code directly without any surrounding explanation or context, and do not enclose it within a code block.".to_string(),
        },
        super::MessageConfig {
          role: "user".to_string(),
          content: "{{ prompt }}\n\n```\n{{ selection }}\n```".to_string(),
        }],
      },
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }
}
