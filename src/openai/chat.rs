use std::{iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;

use crate::chat::Chat;

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIChat {}

impl Chat for OpenAIChat {
  async fn chat(&self, client: Arc<Client>, messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    Ok(iter::empty())
  }
}
