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

#[derive(Debug, Clone, PartialEq)]
pub struct MistralFim {
  pub url: String,
  pub api_key: String,
  pub model: String,
  pub temperature: Option<f64>,
  pub top_p: Option<f64>,
  pub max_tokens: Option<u32>,
  pub min_tokens: Option<u32>,
  pub stop: Vec<String>,
  pub random_seed: Option<u32>,
}

impl Fim for MistralFim {
  async fn fim(
    &self,
    client: Arc<Client>,
    prefix: String,
    suffix: String,
  ) -> anyhow::Result<impl Iterator<Item = String>> {
    let response = client
      .post(&self.url)
      .bearer_auth(&self.api_key)
      .json(&FimRequest {
        model: &self.model,
        prompt: prefix,
        suffix: Some(suffix),
        temperature: self.temperature,
        top_p: self.top_p,
        max_tokens: self.max_tokens,
        min_tokens: self.min_tokens,
        stop: &self.stop,
        random_seed: self.random_seed,
      })
      .send()
      .await?
      .json::<FimResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.message.content))
  }
}
