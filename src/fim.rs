use std::sync::Arc;

use anyhow::Result;
use reqwest::Client;

pub trait Fim {
  async fn fim(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>>;
}
