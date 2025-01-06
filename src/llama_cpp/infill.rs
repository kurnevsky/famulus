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
  pub url: String,
  pub stop: Vec<String>,
}

impl Infill for LlamaCppInfill {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let response = client
      .post(&self.url)
      .json(&InfillRequest {
        input_prefix: prefix,
        input_suffix: suffix,
        stop: &self.stop,
      })
      .send()
      .await?
      .json::<InfillResponse>()
      .await?;

    Ok(iter::once(response.content))
  }
}
