use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::fim::Fim;

#[derive(Clone, PartialEq, Debug, Serialize)]
struct FimRequest<'a> {
  model: &'a str,
  prompt: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  suffix: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  min_tokens: Option<u32>,
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  random_seed: Option<u32>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct Message {
  content: String,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct Choice {
  message: Message,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct FimResponse {
  choices: Vec<Choice>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct MistralFimConfig {
  pub url: String,
  pub model: String,
  pub temperature: Option<f64>,
  pub top_p: Option<f64>,
  pub max_tokens: Option<u32>,
  pub min_tokens: Option<u32>,
  pub stop: Vec<String>,
  pub random_seed: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MistralFim {
  pub api_key: String,
  pub config: MistralFimConfig,
}

impl Fim for MistralFim {
  async fn fim(
    &self,
    client: Arc<Client>,
    prefix: String,
    suffix: String,
  ) -> anyhow::Result<impl Iterator<Item = String>> {
    let response = client
      .post(&self.config.url)
      .bearer_auth(&self.api_key)
      .json(&FimRequest {
        model: &self.config.model,
        prompt: prefix,
        suffix: Some(suffix),
        temperature: self.config.temperature,
        top_p: self.config.top_p,
        max_tokens: self.config.max_tokens,
        min_tokens: self.config.min_tokens,
        stop: &self.config.stop,
        random_seed: self.config.random_seed,
      })
      .send()
      .await?
      .json::<FimResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.message.content))
  }
}
