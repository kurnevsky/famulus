use std::{future::Future, iter, sync::Arc};

use anyhow::Result;
use either::Either;
use reqwest::Client;

pub trait Chat {
  fn chat(
    &self,
    client: Arc<Client>,
    messages: Vec<(String, String)>,
  ) -> impl Future<Output = Result<impl Iterator<Item = String>>> + Send;
}

impl<A: Chat + Sync, B: Chat + Sync> Chat for Either<A, B> {
  async fn chat(&self, client: Arc<Client>, messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    match self {
      Either::Left(a) => a.chat(client, messages).await.map(Either::Left),
      Either::Right(b) => b.chat(client, messages).await.map(Either::Right),
    }
  }
}

impl Chat for () {
  async fn chat(&self, _client: Arc<Client>, _messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    Ok(iter::empty())
  }
}
