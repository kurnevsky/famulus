use std::{iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::infill::Infill;

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

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct OllamaInfillConfig {
  pub url: String,
  pub model: String,
  pub temperature: Option<f64>,
  #[serde(default)]
  pub stop: Vec<String>,
  pub num_predict: Option<u32>,
  pub seed: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OllamaInfill {
  pub api_key: Option<String>,
  pub config: OllamaInfillConfig,
}

impl Infill for OllamaInfill {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.config.url);
    let request = if let Some(ref api_key) = self.api_key {
      request.bearer_auth(api_key)
    } else {
      request
    };
    let response = request
      .json(&GenerateRequest {
        model: &self.config.model,
        prompt: prefix,
        suffix,
        stream: false,
        options: GenerateOptions {
          temperature: self.config.temperature,
          stop: &self.config.stop,
          num_predict: self.config.num_predict,
          seed: self.config.seed,
        },
      })
      .send()
      .await?
      .json::<GenerateResponse>()
      .await?;
    Ok(iter::once(response.response))
  }
}
