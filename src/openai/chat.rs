use std::{iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;

use crate::chat::Chat;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIChatConfig {
  pub url: String,
  pub model: String,
  pub temperature: Option<f64>,
  pub top_p: Option<f64>,
  pub max_tokens: Option<u32>,
  #[serde(default)]
  pub stop: Vec<String>,
  pub seed: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIChat {
  pub config: OpenAIChatConfig,
}

impl Chat for OpenAIChat {
  async fn chat(&self, _client: Arc<Client>, _messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    Ok(iter::empty())
  }
}
