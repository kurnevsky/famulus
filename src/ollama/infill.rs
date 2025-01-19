use std::{env, iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
  config::{ModelConfig, Ollama},
  infill::Infill,
};

#[derive(Clone, PartialEq, Debug, Serialize)]
struct GenerateOptions<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  num_predict: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  seed: Option<u32>,
}

#[derive(Clone, PartialEq, Debug, Serialize)]
struct GenerateRequest<'a> {
  model: &'a str,
  prompt: String,
  suffix: String,
  stream: bool,
  options: GenerateOptions<'a>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct GenerateResponse {
  response: String,
}

impl Infill for ModelConfig<Ollama> {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.url);
    let request = if let Some(ref api_key_env) = self.api_key_env {
      request.bearer_auth(&env::var(api_key_env)?)
    } else {
      request
    };
    let response = request
      .json(&GenerateRequest {
        model: &self.generation_config.model,
        prompt: prefix,
        suffix,
        stream: false,
        options: GenerateOptions {
          temperature: self.generation_config.temperature,
          stop: &self.generation_config.stop,
          num_predict: self.generation_config.max_tokens,
          seed: self.generation_config.seed,
        },
      })
      .send()
      .await?
      .json::<GenerateResponse>()
      .await?;
    Ok(iter::once(response.response))
  }
}
