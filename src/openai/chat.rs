use std::sync::Arc;

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::chat::Chat;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
struct OpenAIChatMessage {
  role: String,
  content: String,
}

#[derive(Clone, PartialEq, Debug, Serialize)]
struct OpanAIChatRequest<'a> {
  model: &'a str,
  messages: Vec<OpenAIChatMessage>,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  seed: Option<u32>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAIChatChoice {
  message: OpenAIChatMessage,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAIChatResponse {
  choices: Vec<OpenAIChatChoice>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIChatConfig {
  pub url: String,
  pub model: String,
  pub temperature: Option<f64>,
  pub top_p: Option<f64>,
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub stop: Vec<String>,
  pub seed: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIChat {
  pub api_key: Option<String>,
  pub config: OpenAIChatConfig,
}

impl Chat for OpenAIChat {
  async fn chat(&self, client: Arc<Client>, messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.config.url);
    let request = if let Some(ref api_key) = self.api_key {
      request.bearer_auth(api_key)
    } else {
      request
    };
    let response = request
      .json(&OpanAIChatRequest {
        model: &self.config.model,
        messages: messages
          .into_iter()
          .map(|(role, content)| OpenAIChatMessage { role, content })
          .collect(),
        temperature: self.config.temperature,
        top_p: self.config.top_p,
        max_tokens: self.config.max_tokens,
        stop: self.config.stop.clone(),
        seed: self.config.seed,
      })
      .send()
      .await?
      .json::<OpenAIChatResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.message.content))
  }
}
