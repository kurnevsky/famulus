use std::{future::Future, iter, sync::Arc};

use anyhow::Result;
use reqwest::Client;

pub trait Chat {
  fn chat(
    &self,
    client: Arc<Client>,
    messages: Vec<(String, String)>,
  ) -> impl Future<Output = Result<impl Iterator<Item = String>>> + Send;
}

impl Chat for () {
  async fn chat(&self, _client: Arc<Client>, _messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    Ok(iter::empty())
  }
}
