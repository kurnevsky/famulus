use std::{env, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
  config::{Mistral, ModelConfig},
  infill::Infill,
};

#[derive(Clone, PartialEq, Debug, Serialize)]
struct InfillRequest<'a> {
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
  #[serde(skip_serializing_if = "Vec::is_empty")]
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
struct InfillResponse {
  choices: Vec<Choice>,
}

impl Infill for ModelConfig<Mistral> {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let response = client
      .post(&self.url)
      .bearer_auth(&env::var(&self.api_key_env)?)
      .json(&InfillRequest {
        model: &self.generation_config.model,
        prompt: prefix,
        suffix: Some(suffix),
        temperature: self.generation_config.temperature,
        top_p: self.generation_config.top_p,
        max_tokens: self.generation_config.max_tokens,
        min_tokens: self.generation_config.min_tokens,
        stop: &self.generation_config.stop,
        random_seed: self.generation_config.seed,
      })
      .send()
      .await?
      .json::<InfillResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.message.content))
  }
}
