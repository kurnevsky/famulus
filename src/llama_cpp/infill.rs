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
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
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
        stop: &self.generation_config.stop,
      })
      .send()
      .await?
      .json::<InfillResponse>()
      .await?;

    Ok(iter::once(response.content))
  }
}
