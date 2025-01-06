use std::{iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::infill::Infill;

#[derive(Clone, PartialEq, Debug, Serialize)]
struct InfillRequest<'a> {
  input_prefix: String,
  input_suffix: String,
  stop: &'a Vec<String>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct InfillResponse {
  content: String,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
pub struct LlamaCppInfillConfig {
  pub url: String,
  pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppInfill {
  pub api_key: Option<String>,
  pub config: LlamaCppInfillConfig,
}

impl Infill for LlamaCppInfill {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.config.url);
    let request = if let Some(ref api_key) = self.api_key {
      request.bearer_auth(api_key)
    } else {
      request
    };
    let response = request
      .json(&InfillRequest {
        input_prefix: prefix,
        input_suffix: suffix,
        stop: &self.config.stop,
      })
      .send()
      .await?
      .json::<InfillResponse>()
      .await?;

    Ok(iter::once(response.content))
  }
}
