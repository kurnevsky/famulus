use std::{env, iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
  config::{LlamaCpp, ModelConfig},
  infill::Infill,
};

#[derive(Clone, PartialEq, Debug, Serialize)]
struct InfillRequest<'a> {
  input_prefix: String,
  input_suffix: String,
  temperature: Option<f64>,
  top_p: Option<f64>,
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  seed: Option<u32>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct InfillResponse {
  content: String,
}

impl Infill for ModelConfig<LlamaCpp> {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.url);
    let request = if let Some(ref api_key_env) = self.api_key_env {
      request.bearer_auth(&env::var(api_key_env)?)
    } else {
      request
    };
    let response = request
      .json(&InfillRequest {
        input_prefix: prefix,
        input_suffix: suffix,
        temperature: self.generation_config.temperature,
        top_p: self.generation_config.top_p,
        max_tokens: self.generation_config.max_tokens,
        stop: &self.generation_config.stop,
        seed: self.generation_config.seed,
      })
      .send()
      .await?
      .json::<InfillResponse>()
      .await?;

    Ok(iter::once(response.content))
  }
}
