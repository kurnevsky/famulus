use std::{iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::fim::Fim;

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

// curl -XPOST "localhost:8081/infill" -d '{"input_prefix": "1, ", "input_suffix": ", 5", "stop": ["<|file_separator|>"]}' -H "Content-Type: application/json"
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppFim {
  pub url: String,
  pub stop: Vec<String>,
}

impl Fim for LlamaCppFim {
  async fn fim(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
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
