use std::{collections::HashMap, fs::File, io::BufReader, sync::Arc};

use anyhow::Result;
use lsp_server::{Connection, Message, RequestId, Response};
use lsp_types::{
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification},
  request::{InlineCompletionRequest, Request, Shutdown},
  CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
  InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse, NumberOrString, OneOf, Range,
  ServerCapabilities, TextDocumentSyncKind, Uri,
};
use ropey::Rope;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

#[derive(Debug, Default)]
struct State {
  documents: HashMap<Uri, Rope>,
  tasks: Arc<papaya::HashMap<RequestId, JoinHandle<()>>>,
}

#[derive(Clone, PartialEq, Debug, Serialize)]
struct FimRequest {
  model: String,
  prompt: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  suffix: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  min_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  stop: Option<String>, // TODO: array?
  #[serde(skip_serializing_if = "Option::is_none")]
  random_seed: Option<u32>,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct Message_ {
  content: String,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct Choice {
  message: Message_,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct FimResponse {
  choices: Vec<Choice>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (connection, io_threads) = Connection::stdio();
  let server_capabilities = ServerCapabilities {
    inline_completion_provider: Some(OneOf::Left(true)),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    ..Default::default()
  };
  let _initialization_args = connection.initialize(serde_json::to_value(server_capabilities)?)?;

  let sender = Arc::new(connection.sender);
  let client = Arc::new(reqwest::Client::new());

  let mut state = State::default();

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == Shutdown::METHOD {
          return Ok(());
        } else if request.method == InlineCompletionRequest::METHOD {
          let (request_id, params) = request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD)?;
          let rope = state
            .documents
            .get(&params.text_document_position.text_document.uri)
            .expect("missing document");

          let index = rope.line_to_char(params.text_document_position.position.line as usize)
            + params.text_document_position.position.character as usize;
          let prompt = rope.slice(0..index).to_string();
          let suffix = rope.slice(index..rope.len_chars()).to_string();

          let client = client.clone();
          let sender = sender.clone();
          let tasks = state.tasks.clone();
          let request_id_c = request_id.clone();
          let future = async move {
            let response = client
              .post("https://api.mistral.ai/v1/fim/completions")
              .bearer_auth("")
              .json(&FimRequest {
                model: "codestral-latest".to_string(),
                prompt,
                suffix: Some(suffix),
                temperature: Some(0.0),
                top_p: None,
                max_tokens: None,
                min_tokens: None,
                stop: None,
                random_seed: None,
              })
              .send()
              .await;
            match response {
              Result::Ok(response) => {
                let response: FimResponse = response.json().await.unwrap(); // TODO
                let range = Range::new(
                  params.text_document_position.position,
                  params.text_document_position.position,
                );
                let completion_items = response
                  .choices
                  .into_iter()
                  .map(|choice| InlineCompletionItem {
                    insert_text: choice.message.content,
                    filter_text: None,
                    range: Some(range),
                    command: None,
                    insert_text_format: None,
                  })
                  .collect();
                tasks.pin().remove(&request_id_c);
                if let Result::Err(error) = sender.send(Message::Response(Response::new_ok(
                  request_id_c,
                  InlineCompletionResponse::Array(completion_items),
                ))) {
                  log::warn!("Failed to send server response: {}", error);
                }
              }
              Result::Err(error) => {
                tasks.pin().remove(&request_id_c);
                log::warn!("Failed to get mistral response: {}", error);
              }
            }
          };
          let handle = tokio::task::spawn(future);
          state.tasks.pin().insert(request_id, handle);
        }
      }
      Message::Notification(notification) => {
        if notification.method == DidOpenTextDocument::METHOD {
          let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
          let file = File::open(params.text_document.uri.path().as_str())?;
          let reader = BufReader::new(file);
          let rope = Rope::from_reader(reader)?;
          state.documents.insert(params.text_document.uri, rope);
        } else if notification.method == DidCloseTextDocument::METHOD {
          let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
          state.documents.remove(&params.text_document.uri);
        } else if notification.method == DidChangeTextDocument::METHOD {
          let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
          for change in params.content_changes {
            if let Some(range) = change.range {
              let rope = state
                .documents
                .get_mut(&params.text_document.uri)
                .expect("missing document");
              let start_index = rope.line_to_char(range.start.line as usize) + range.start.character as usize;
              let end_index = rope.line_to_char(range.end.line as usize) + range.end.character as usize;
              rope.remove(start_index..end_index);
              rope.insert(start_index, &change.text);
            } else {
              let rope = Rope::from_str(&change.text);
              state.documents.insert(params.text_document.uri.clone(), rope);
            }
          }
        } else if notification.method == Cancel::METHOD {
          let params: CancelParams = serde_json::from_value(notification.params)?;
          let id: RequestId = match params.id {
            NumberOrString::Number(id) => id.into(),
            NumberOrString::String(id) => id.into(),
          };
          if let Some(handle) = state.tasks.pin().remove(&id) {
            handle.abort();
          }
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}
