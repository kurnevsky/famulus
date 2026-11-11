use std::sync::Arc;

use ramhorns::Template;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct GenerationConfig {
  pub model: Option<String>,
  #[serde(default)]
  pub temperature: Option<f64>,
  #[serde(default)]
  pub top_p: Option<f64>,
  #[serde(default)]
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub min_tokens: Option<u32>,
  #[serde(default)]
  pub stop: Vec<String>,
  #[serde(default)]
  pub seed: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
  pub url: String,
  #[serde(default)]
  pub api_key_env: Option<String>,
  #[serde(flatten)]
  pub generation_config: GenerationConfig,
}

#[derive(Debug)]
pub struct TemplateConfig(pub Template<'static>);

impl PartialEq for TemplateConfig {
  fn eq(&self, other: &Self) -> bool {
    self.0.source() == other.0.source()
  }
}

impl<'de> Deserialize<'de> for TemplateConfig {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let s = String::deserialize(deserializer)?;
    let template = Template::new(s).map_err(|e| serde::de::Error::custom(e.to_string()))?;
    Ok(TemplateConfig(template))
  }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CopilotConfig {
  pub api_key_env: String,
  #[serde(default)]
  pub temperature: Option<f64>,
  #[serde(default)]
  pub top_p: Option<f64>,
  #[serde(default)]
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub stop: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(tag = "provider")]
pub enum CompletionConfig {
  #[default]
  Empty,
  Mistral {
    config: Arc<ModelConfig>,
  },
  LlamaCpp {
    config: Arc<ModelConfig>,
  },
  Ollama {
    config: Arc<ModelConfig>,
  },
  OpenAICompletions {
    config: Arc<ModelConfig>,
    template: Arc<TemplateConfig>,
  },
  Copilot {
    config: Arc<CopilotConfig>,
  },
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
  #[serde(default)]
  pub infill: CompletionConfig,
}

impl Config {
  pub fn get_infill(&self) -> crate::infill::InfillProvider {
    self.infill.clone().into()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn mistral_config() {
    let s = r#"{"infill":{"provider":"Mistral","config":{"url":"https://api.mistral.ai/v1/fim/completions","api_key_env":"MISTRAL_API_KEY","model":"codestral-latest","temperature":0.7,"top_p":0.95,"max_tokens":1024,"min_tokens":1,"stop":["\n\n"],"seed":42}}}"#;
    let _config: Config = serde_json::from_str(s).unwrap();
  }

  #[test]
  fn copilot_config() {
    let s = r#"{"infill":{"provider":"Copilot","config":{"api_key_env":"COPILOT_TOKEN","temperature":0.0,"max_tokens":1024,"stop":["\n\n"]}}}"#;
    let _config: Config = serde_json::from_str(s).unwrap();
  }

  #[test]
  fn empty_config() {
    let s = r#"{}"#;
    let _config: Config = serde_json::from_str(s).unwrap();
  }
}
