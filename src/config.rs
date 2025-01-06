use serde::Deserialize;

use crate::{llama_cpp::infill::LlamaCppInfillConfig, mistral::infill::MistralInfillConfig};

#[derive(Clone, PartialEq, Debug, Deserialize)]
#[serde(tag = "privider", content = "config")]
pub enum CompletionConfig {
  Mistral(MistralInfillConfig),
  LlamaCpp(LlamaCppInfillConfig),
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct Config {
  infill: CompletionConfig,
}

#[cfg(test)]
mod tests {
  use crate::{llama_cpp::infill::LlamaCppInfillConfig, mistral::infill::MistralInfillConfig};

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
          "url": "http://localhost:8080/v1/infill",
          "stop": ["<|file_separator|>"]
        }
      }
    }
    "#;
    let config = Config {
      infill: CompletionConfig::LlamaCpp(LlamaCppInfillConfig {
        url: "http://localhost:8080/v1/infill".to_string(),
        stop: vec!["<|file_separator|>".to_string()],
      }),
    };
    let parsed: Config = serde_json::from_str(str).unwrap();
    assert_eq!(parsed, config);
  }
}
